import os
import threading
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .model import Detector, spectrum
from .imaging import decode, tensor_image, data_url, gradcam, overlay, MAX_BYTES

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title='Fake Catcher')
app.mount('/static', StaticFiles(directory=ROOT / 'web'), name='static')
lock = threading.Lock()
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def load_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt.get('schema') != 1 or ckpt.get('labels') != {'real': 0, 'fake': 1}:
        raise ValueError('지원하지 않는 체크포인트 형식입니다.')
    if ckpt.get('image_size') != 256 or ckpt.get('trained_steps', 0) < 1:
        raise ValueError('학습 완료 체크포인트가 필요합니다.')
    model = Detector(ckpt['variant'])
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.dataset_note = ckpt.get('dataset_note', '')
    model.eval()
    return model


checkpoint = Path(os.environ.get('FAKE_CATCHER_CHECKPOINT', str(ROOT / 'models' / 'best.pt')))
model = None
load_error = None
if checkpoint.is_file():
    try:
        model = load_checkpoint(checkpoint)
    except Exception:
        load_error = '가중치를 읽지 못했습니다. 체크포인트 형식을 확인하세요.'


@app.get('/')
def home():
    return FileResponse(ROOT / 'web' / 'index.html')


@app.get('/api/status')
def status():
    return {'ready': model is not None, 'variant': model.variant if model else None,
            'message': load_error or ('CIFAKE 예비 학습 모델 · 최신 생성기 성능 미검증' if model and 'CIFAKE' in model.dataset_note else '분석 준비 완료' if model else '학습된 모델 대기 중 · FFT 미리보기 사용 가능')}


def analyze(raw):
    try:
        image = decode(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    batch = tensor_image(image).unsqueeze(0)
    fft = spectrum(batch)[0, 0].numpy()
    result = {'ready': model is not None, 'original_size': list(image.size),
              'input': data_url(image.resize((256, 256))),
              'spectrum': data_url(Image.fromarray(np.uint8(fft * 255))),
              'score': None, 'heatmap': None, 'label': '판정 대기', 'elapsed_ms': None}
    if model is not None:
        with lock:
            started = time.perf_counter()
            score, cam, active = gradcam(model, batch)
            elapsed = (time.perf_counter() - started) * 1000
        result.update(score=score, label='AI 생성 쪽' if score >= .5 else '실제 사진 쪽',
                      heatmap=data_url(overlay(image, cam)), heatmap_active=active,
                      elapsed_ms=round(elapsed, 1), variant=model.variant,
                      dataset_note=getattr(model, 'dataset_note', ''))
    return result


@app.post('/api/analyze')
async def analyze_endpoint(request: Request):
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, '이미지는 10MB 이하로 올려주세요.')
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(analyze, bytes(raw))
