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
            x0, y0 = df[['Longitude (decimal)', 'Latitude (decimal)']].min()
            s = (
                df[['Longitude (decimal)', 'Latitude (decimal)']].max()
                - (x0, y0)
            ).max()
            fns = {
                'Longitude': lambda x: (100/s)*(x - x0),
                'Latitude': lambda y: (100/s)*(s + y0 - y)
            }
            for t in ['Longitude', 'Latitude']:
                source = f'{t} (decimal)'
                dest = f'{t} (relative)'
                df[dest] = fns[t](df[source])
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
