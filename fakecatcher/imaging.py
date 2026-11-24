import base64
import io
import warnings
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

SIZE = 256
MAX_BYTES = 10 * 1024 * 1024


def decode(raw):
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError('이미지는 10MB 이하로 올려주세요.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format not in {'JPEG', 'PNG', 'WEBP'}:
                    raise ValueError('JPG, PNG, WEBP 이미지만 지원합니다.')
                if im.width * im.height > 20_000_000:
                    raise ValueError('이미지는 2천만 픽셀 이하로 올려주세요.')
                return ImageOps.exif_transpose(im).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('읽을 수 없는 이미지입니다.') from exc


def tensor_image(image):
    # Same geometric preprocessing for train/eval/web. UI shows this exact model input.
    image = image.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)


def data_url(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def gradcam(model, batch):
    """Grad-CAM for the predicted class, from RGB spatial features (not FFT bins)."""
    with torch.enable_grad():
        logits, features = model(batch, return_features=True)
        fake = logits[0].item() >= 0
        score = logits[0] if fake else -logits[0]
        gradients = torch.autograd.grad(score, features)[0]
        cam = (gradients.mean((-2, -1), keepdim=True) * features).sum(1, keepdim=True).relu()
        cam = torch.nn.functional.interpolate(cam, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        maximum = float(cam.max())
        return float(logits[0].sigmoid().detach()), cam / max(maximum, 1e-8), maximum > 1e-8


def overlay(image, cam):
    original = np.asarray(image.resize((SIZE, SIZE)), dtype=np.float32) / 255
    heat = np.stack([np.clip(cam * 2, 0, 1), np.clip(2 - abs(cam * 4 - 2), 0, 1), 1 - cam], -1)
    alpha = (cam * .55)[..., None]
    return Image.fromarray(np.uint8(np.clip(original * (1 - alpha) + heat * alpha, 0, 1) * 255))
