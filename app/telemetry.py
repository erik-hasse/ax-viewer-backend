import base64
import io
import json
from pathlib import Path
import requests

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.spatial import ConvexHull


def get_cumulative_time(df):
    cumulative_lap_times = (
        df.groupby('Lap')['Elapsed Time (ms)']
        .max().cumsum().to_dict()
    )
    cumulative_lap_times = {
        **{k+1: v for k, v in cumulative_lap_times.items()},
        0: 0
    }
    return (
        df['Elapsed Time (ms)']
        + df['Lap'].map(cumulative_lap_times)
    )


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.
    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def rotate(p, origin=(0, 0), angle=0):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def best_rotation(p):
    corners = minimum_bounding_rectangle(p)
    p0 = corners[corners[:, 0].argmin()]
    p1 = corners[corners[:, 1].argmin()]
    p2 = corners[corners[:, 0].argmax()]
    left_legs = p0-p1
    right_legs = p2-p1
    if np.sqrt(np.sum(left_legs**2)) < np.sqrt(np.sum(right_legs**2)):
        theta = np.pi/2 - np.arctan2(*left_legs[::-1])
    else:
        theta = np.pi/2 - np.arctan2(*right_legs[::-1])
    print(theta)
    return rotate(p, angle=theta).T


class Telemetry:
    def __init__(self, telem_path, video_path, local_path, make_plot=True):
        self._local_path = Path(local_path)

        if video_path.startswith('http'):
            self._video_path = self._local_path / video_path.split('/')[-1]
            if not self._video_path.exists():
                self._video_path.write_bytes(requests.get(video_path).content)
        else:
            self._video_path = Path(video_path)

        if telem_path.startswith('http'):
            self._telem_path = self._local_path / telem_path.split('/')[-1]
            if not self._telem_path.exists():
                self._telem_path.write_bytes(requests.get(telem_path).content)
        else:
            self._telem_path = Path(telem_path)

        self._video_name = self._video_path.stem

        self._param_path = (
            self._local_path / (self._video_name + '.json')
        )
        if not self._param_path.exists():
            json.dump({}, self._param_path.open('w'))

        d = json.load(self._param_path.open('r'))

        self._frame_length = d.get('frame_length')
        self._best_offset = d.get('best_offset')
        self._diffs = None
        self._df = None
        self._best_fit_plot = None
        if make_plot:
            self.best_fit_plot

    @property
    def frame_length(self):
        if self._frame_length is None:
            cap = cv2.VideoCapture(str(self._video_path))
            self._frame_length = 1 / cap.get(cv2.CAP_PROP_FPS)
            self.update_json()
        return self._frame_length

    @property
    def df(self):
        if self._df is None:
            df = pd.read_csv(self._telem_path)
            df['Total Elapsed Time (ms)'] = get_cumulative_time(df)
            try:
                rotated = best_rotation(
                    df[
                        ['Longitude (decimal)', 'Latitude (decimal)']
                    ].to_numpy()
                )
            except Exception:
                rotated = df[
                    ['Longitude (decimal)', 'Latitude (decimal)']
                ].to_numpy().T
            x0, y0 = rotated.min(axis=1)
            s = (rotated.max(axis=1) - (x0, y0)).max()
            fns = {
                'Longitude': lambda x: (100/s)*(x - x0),
                'Latitude': lambda y: (100/s)*(s + y0 - y)
            }
            for i, t in enumerate(['Longitude', 'Latitude']):
                dest = f'{t} (relative)'
                df[dest] = fns[t](rotated[i])
                df[dest] -= df[dest].min()
                df[dest] += 5

            self._df = df[~df['Total Elapsed Time (ms)'].duplicated()]
        return self._df

    @property
    def best_offset(self):
        if self._best_offset is None:
            o = scipy.optimize.brute(
                lambda x: 1-self.test_offset(x[0])[0],
                ranges=[(-1000, 5000)],
                Ns=11,
                finish=None
            )
            o = scipy.optimize.brute(
                lambda x: 1-self.test_offset(x[0])[0],
                ranges=[(o-600, o+600)],
                finish=None
            )
            self._best_offset = scipy.optimize.brute(
                lambda x: 1-self.test_offset(x[0])[0],
                ranges=[(o-240, o+240)],
                Ns=11,
                finish=None
            )
            self.update_json()
        return self._best_offset

    @property
    def diffs(self):
        if self._diffs is None:
            self._diffs = self._get_video_diffs()
        return self._diffs

    def update_json(self):
        json.dump(
            {
                'frame_length': self._frame_length,
                'best_offset': self._best_offset
            },
            self._param_path.open('w')
        )

    def _get_video_diffs(self):
        cap = cv2.VideoCapture(str(self._video_path))
        ret, current_frame = cap.read()
        previous_frame = current_frame
        diffs = []
        while(cap.isOpened()):
            diff = cv2.absdiff(current_frame, previous_frame)

            diffs.append(np.sqrt((diff*diff).mean()))
            previous_frame = current_frame.copy()
            ret, current_frame = cap.read()
            if not ret:
                break
        cap.release()

        return diffs

    def test_offset(self, offset):
        tdf = self.df.set_index('Total Elapsed Time (ms)')
        train = (
            tdf.iloc[pd.Index([
                i*self.frame_length*1000+offset
                for i in range(1, len(self.diffs))
            ]).map(lambda x: tdf.index.get_loc(x, 'nearest'))
            ][['Speed (MPH)', 'Steering Angle (deg)']]
            .assign(diffs=self.diffs[1:], const=1)
        )

        train = train[~train.index.duplicated()]
        train['Steering Angle (deg)'] = train['Steering Angle (deg)'].abs()

        X = train[['Speed (MPH)', 'Steering Angle (deg)', 'const']].to_numpy()
        y = train['diffs'].to_numpy()
        A, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - res / (y.size * y.var())
        return r2[0], train, np.dot(X, A.reshape(-1, 1))

    @property
    def best_fit_plot(self):
        if self._best_fit_plot is not None:
            return self._best_fit_plot
        fig, [ax, ax1] = plt.subplots(nrows=2, figsize=(9, 12))
        _, train, preds = self.test_offset(self.best_offset)
        train.index = train.index / 1000

        fig.suptitle('Best fit plot', fontsize=16)
        ax.set_title('Frame difference')
        ax.plot(train['diffs'])
        ax.plot(train.index, preds)
        ax.set_ylabel('RMS frame difference')

        ax1.sharex(ax)
        ax1.set_title('Features')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Speed (MPH)')
        ax1.plot(train['Speed (MPH)'])

        ax2 = ax1.twinx()
        ax2.set_ylabel('abs(Steering angle) (deg)')
        ax2.plot(train['Steering Angle (deg)'], color='orange')

        fig.tight_layout()

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        self._best_fit_plot = base64.b64encode(img.read())
        return self._best_fit_plot.decode()
