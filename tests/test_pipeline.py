import io
import sys
import numpy as np
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient
from fakecatcher.model import Detector, spectrum
from fakecatcher.imaging import gradcam, decode, MAX_BYTES
from fakecatcher import server
from fakecatcher.train import Images, check_overlap, main

torch.set_num_threads(2)


def png(value=120):
    stream = io.BytesIO()
    Image.new('RGB', (72, 48), (value, 70, 20)).save(stream, format='PNG')
    return stream.getvalue()


def test_fft_constant_and_sinusoid():
    assert spectrum(torch.ones(1, 3, 256, 256)).max() == 0
    x = torch.arange(256)
    signal = (.5 + .4 * torch.cos(2 * torch.pi * 20 * x / 256)).view(1, 1, 1, 256).expand(1, 3, 256, 256)
    value = spectrum(signal)[0, 0]
    assert value[128, 108] > .99 and value[128, 148] > .99
    assert torch.isfinite(value).all()


@pytest.mark.parametrize('variant', ['rgb', 'rgb_fft'])
def test_model_gradient_and_heatmap(variant):
    model = Detector(variant).eval()
    batch = torch.rand(2, 3, 256, 256)
    logits = model(batch)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])).backward()
    assert model.spatial[0][0].weight.grad.abs().sum() > 0
    if model.frequency is not None:
        assert model.frequency[0].weight.grad.abs().sum() > 0
    before = {k: v.clone() for k, v in model.state_dict().items()}
    score, cam, _ = gradcam(model, batch[:1])
    assert 0 <= score <= 1 and cam.shape == (256, 256) and np.isfinite(cam).all()
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_preview_does_not_invent_prediction(monkeypatch):
    monkeypatch.setattr(server, 'model', None)
    client = TestClient(server.app)
    assert client.get('/').status_code == 200
    assert client.get('/api/status').json()['ready'] is False
    response = client.post('/api/analyze', content=png())
    assert response.status_code == 200
    data = response.json()
    assert data['score'] is None and data['heatmap'] is None
    assert data['spectrum'].startswith('data:image/png;base64,')


def test_invalid_uploads():
    client = TestClient(server.app)
    assert client.post('/api/analyze', content=b'not an image').status_code == 400
    assert client.post('/api/analyze', content=b'x' * (MAX_BYTES + 1)).status_code == 413
    stream = io.BytesIO()
    Image.new('RGB', (5001, 4000)).save(stream, format='PNG')
    with pytest.raises(ValueError, match='픽셀'):
        decode(stream.getvalue())


def test_split_leakage_rejected(tmp_path):
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            (folder / 'same.png').write_bytes(png())
    with pytest.raises(ValueError, match='동일'):
        check_overlap(Images(tmp_path / 'train'), Images(tmp_path / 'val'))


def test_train_checkpoint_and_inference(tmp_path, monkeypatch):
    # Random fixtures verify plumbing only, never measure detector accuracy.
    rng = np.random.default_rng(42)
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            folder = tmp_path / 'data' / split / label
            folder.mkdir(parents=True)
            for i in range(2):
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(folder / f'{i}.png')
    output = tmp_path / 'training'
    monkeypatch.setattr(sys, 'argv', ['train', '--data', str(tmp_path / 'data'), '--output', str(output), '--epochs', '1', '--batch-size', '2', '--no-pretrained', '--device', 'cpu'])
    main()
    checkpoint = output / 'best.pt'
    trained = server.load_checkpoint(checkpoint)
    monkeypatch.setattr(server, 'model', trained)
    result = TestClient(server.app).post('/api/analyze', content=png()).json()
    assert result['ready'] and 0 <= result['score'] <= 1
    assert result['heatmap'].startswith('data:image/png;base64,')
    assert result['elapsed_ms'] > 0
    monkeypatch.setattr(sys, 'argv', ['train', '--data', str(tmp_path / 'data'), '--output', str(tmp_path / 'evaluation'), '--evaluate', str(checkpoint)])
    main()
    assert (tmp_path / 'evaluation' / 'test.json').is_file()
    # Resumed training must match uninterrupted training, including optimizer/RNG.
    common = ['train', '--data', str(tmp_path / 'data'), '--epochs', '2', '--batch-size', '2', '--no-pretrained', '--device', 'cpu']
    monkeypatch.setattr(sys, 'argv', common + ['--output', str(output), '--resume', str(output / 'last.pt')])
    main()
    uninterrupted = tmp_path / 'uninterrupted'
    monkeypatch.setattr(sys, 'argv', common + ['--output', str(uninterrupted)])
    main()
    resumed = torch.load(output / 'last.pt', weights_only=True)
    continuous = torch.load(uninterrupted / 'last.pt', weights_only=True)
    assert resumed['epoch'] == 2 and resumed['trained_steps'] == continuous['trained_steps']
    assert all(torch.equal(value, continuous['state_dict'][key]) for key, value in resumed['state_dict'].items())
    monkeypatch.setattr(sys, 'argv', common + ['--output', str(output), '--resume', str(output / 'last.pt'), '--lr', '.01'])
    with pytest.raises(ValueError, match='configuration mismatch'):
        main()


def test_untrained_checkpoint_rejected(tmp_path):
    checkpoint = tmp_path / 'bad.pt'
    torch.save({'schema': 1, 'labels': {'real': 0, 'fake': 1}, 'image_size': 256, 'trained_steps': 0}, checkpoint)
    with pytest.raises(ValueError, match='학습 완료'):
        server.load_checkpoint(checkpoint)
