from pathlib import Path
import requests
import urllib

import bs4
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
import paramiko

from app.telemetry import Telemetry

app = FastAPI()

origins = [
    'http://localhost:3000',
    'localhost:3000',
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


def get_data(prod=True):
    if prod:
        base_url = 'http://teslausb.local/static_files/'
        data = [f['name'] for f in requests.get(base_url).json()]
    else:
        base_url = 'http://localhost:8000/'
        soup = bs4.BeautifulSoup(
            requests.get(base_url).content, features='lxml'
        )
        data = [x.contents[0] for x in soup.select('a')]

    telemetry_files = sorted(
        base_url + f for f in data
        if f.endswith('.csv') and f.startswith('telemetry-v1')
    )
    video_files = sorted(
        base_url + f for f in data
        if f.endswith('.mp4') and f.startswith('laps')
    )

    # Assume for now now that there are an equal number of each file
    return {
        f[1].removeprefix(base_url + 'laps-').removesuffix('.mp4'): f
        for f in zip(telemetry_files, video_files)
    }


static = Path('/app/static')
telemetry = {
    k: Telemetry(*p, local_path=static)
    for k, p in get_data().items()
}


@app.get('/', tags=['root'])
async def read_root() -> dict:
    return {'message': 'Track mode viewer.'}


@app.get('/force_sync', tags=['sync'])
async def force_sync() -> dict:
    # Force a sync from the car
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('teslausb.local', username='pi')
    stdin, stdout, stderr = ssh.exec_command('sudo /root/bin/force_sync.sh')
    stdout.channel.settimeout(60)
    stdout.channel.recv_exit_status()
    ssh.close()
    return {'message': 'Syncing'}


@app.get('/telemetry', tags=['telemetry'])
async def get_timestamps(background_tasks: BackgroundTasks) -> dict:
    new_data = get_data()
    telemetry.update({
        k: Telemetry(*p, local_path=static)
        for k, p in new_data.items()
        if k not in telemetry
    })
    return {'data': list(new_data.keys())}


@app.get('/telemetry/{timestamp}', tags=['telemetry'])
async def get_telemetry(timestamp: str) -> dict:
    df = telemetry[urllib.parse.unquote_plus(timestamp)].df
    return df.to_dict()


@app.get('/fit_plot/{timestamp}', tags=['telemetry'])
async def get_plots(timestamp: str) -> dict:
    t = telemetry[urllib.parse.unquote_plus(timestamp)]
    return {'img': t.best_fit_plot}


@app.get('/telemetry/{timestamp}/{telemetry_time}', tags=['telemetry'])
async def get_telemetry_at_time(timestamp: str, telemetry_time: int) -> dict:
    t = telemetry[urllib.parse.unquote_plus(timestamp)]
    df = t.df
    df = df[
        df['Total Elapsed Time (ms)'] <= max(
            telemetry_time + t.best_offset, df['Total Elapsed Time (ms)'][0]
        )
    ]
    return df.iloc[-1].to_dict()


@app.get('/map/{timestamp}', tags=['telemetry'])
async def get_map(timestamp: str) -> dict:
    df = telemetry[urllib.parse.unquote_plus(timestamp)].df
    df = df[df['Lap'] == 0]
    return {
        k.lower(): df[f'{k} (relative)'].to_list()
        for k in ['Longitude', 'Latitude']
    }
