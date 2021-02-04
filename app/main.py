from pathlib import Path
import urllib

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


def get_data(path):
    path = Path(path)
    telemetry_files = sorted([
        p for p in path.iterdir()
        if p.suffix == '.csv' and p.stem.startswith('telemetry-v1')
    ])
    video_files = sorted([
        p for p in path.iterdir()
        if p.suffix == '.mp4' and p.stem.startswith('laps')
    ])
    # Assume for now now that there are an equal number of each file
    return {
        f[1].stem.removeprefix('laps-'): f
        for f in zip(telemetry_files, video_files)
    }


static = Path('/app/static')
telemetry = {
    k: Telemetry(*p)
    for k, p in get_data(static).items()
}


@app.get('/', tags=['root'])
async def read_root() -> dict:
    return {'message': 'Track mode viewer.'}


@app.get('/telemetry', tags=['telemetry'])
async def get_timestamps(background_tasks: BackgroundTasks) -> dict:
    new_data = get_data(static)
    telemetry.update({
        k: Telemetry(*p)
        for k, p in new_data.items()
        if k not in telemetry
    })
    return {'data': list(new_data.keys())}


@app.get('/telemetry/{timestamp}', tags=['telemetry'])
async def get_telemetry(timestamp: str) -> dict:
    df = telemetry[urllib.parse.unquote_plus(timestamp)].df
    return df.to_dict()


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
    return {
        k.lower(): df[f'{k} (relative)'].to_list()
        for k in ['Longitude', 'Latitude']
    }
