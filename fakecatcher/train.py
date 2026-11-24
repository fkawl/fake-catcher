"""python -m fakecatcher.train --data data --output runs/rgb_fft --variant rgb_fft"""
import argparse
import hashlib
import io
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from .model import Detector
from .imaging import tensor_image


class Images(Dataset):
    def __init__(self, folder, augment=False):
        self.samples = []
        self.augment = augment
        for name, label in [('real', 0), ('fake', 1)]:
            paths = sorted(p for p in (Path(folder) / name).rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
            if not paths:
                raise ValueError(f'{folder}/{name}: 이미지가 없습니다.')
            self.samples += [(p, label) for p in paths]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert('RGB')
        if self.augment:
            if random.random() < .5:
                image = ImageOps.mirror(image)
            if random.random() < .5:
                stream = io.BytesIO()
                image.save(stream, format='JPEG', quality=random.randint(60, 95))
                stream.seek(0)
                with Image.open(stream) as compressed:
                    image = compressed.convert('RGB')
            if random.random() < .2:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(.1, 1)))
        return tensor_image(image), torch.tensor(label, dtype=torch.float32)


def check_overlap(left, right):
    # Detect identical decoded pixels too, even when metadata/file names differ.
    def fingerprint(path):
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            return hashlib.sha256(str(im.size).encode() + im.tobytes()).digest()
    hashes = {fingerprint(p) for p, _ in left.samples}
    if any(fingerprint(p) in hashes for p, _ in right.samples):
        raise ValueError('학습/평가 데이터에 동일 이미지가 있습니다. 분리 후 재실행하세요.')


def metrics(labels, scores):
    predictions = np.asarray(scores) >= .5
    return {'auroc': float(roc_auc_score(labels, scores)),
            'f1': float(f1_score(labels, predictions, zero_division=0)),
            'accuracy': float(np.mean(predictions == np.asarray(labels))),
            'confusion_matrix_real_fake': confusion_matrix(labels, predictions, labels=[0, 1]).tolist()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, scores = [], []
    for images, targets in loader:
        scores.extend(model(images.to(device)).sigmoid().cpu().tolist())
        labels.extend(targets.tolist())
    return metrics(labels, scores)


def main():
    parser = argparse.ArgumentParser(description='RGB / RGB+FFT 학습 및 평가')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['rgb', 'rgb_fft'], default='rgb_fft')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-pretrained', action='store_true', help='오프라인 구조 검증용. 일반 학습은 사전학습 사용 권장')
    parser.add_argument('--evaluate', type=Path, help='체크포인트를 --data/test에서 평가하고 종료')
    parser.add_argument('--resume', type=Path, help='last.pt로 중단된 학습 재개; epochs는 최종 목표 횟수')
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs >= 1, batch-size >= 2가 필요합니다.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if args.evaluate:
        ckpt = torch.load(args.evaluate, map_location='cpu', weights_only=True)
        model = Detector(ckpt['variant']).to(device)
        model.load_state_dict(ckpt['state_dict'])
        dataset = Images(args.data / 'test')
        result = evaluate(model, DataLoader(dataset, batch_size=args.batch_size), device)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'test.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
        return
    training, validation = Images(args.data / 'train', True), Images(args.data / 'val')
    check_overlap(training, validation)
    args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    # Keep downloaded weights inside this project/run, never in the user's global cache.
    torch.hub.set_dir(str(args.output / 'pretrained-cache'))
    model = Detector(args.variant, pretrained=not args.no_pretrained and not args.resume).to(device)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    history, best, steps, start = [], -1, 0, 0
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    fingerprint = hashlib.sha256()
    windows_fingerprint = hashlib.sha256()
    for dataset in [training, validation]:
        for path, label in dataset.samples:
            relative = path.relative_to(args.data).as_posix()
            content_hash = hashlib.sha256(path.read_bytes()).digest()
            for digest, spelling in [(fingerprint, relative), (windows_fingerprint, relative.replace('/', '\\'))]:
                digest.update(spelling.encode())
                digest.update(str(label).encode())
                digest.update(content_hash)
    data_id = fingerprint.hexdigest()
    note_file = args.data / 'dataset.json'
    dataset_note = json.loads(note_file.read_text(encoding='utf-8')).get('note', '') if note_file.exists() else 'User dataset; external generalization not verified.'
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        for key in ['variant', 'batch_size', 'seed', 'lr']:
            if saved['config'][key] != config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        if saved['data_id'] not in {data_id, windows_fingerprint.hexdigest()}:
            raise ValueError('Resume dataset differs from original run')
        if args.resume.resolve().parent != args.output.resolve():
            raise ValueError('Resume output must be the original run directory')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        history, best, steps, start = saved['history'], saved['best'], saved['trained_steps'], saved['epoch']
        torch.set_rng_state(saved['torch_rng'])
        random.setstate(saved['python_rng'])
        npstate = saved['numpy_rng']
        np.random.set_state((npstate[0], np.array(npstate[1], dtype=np.uint32), *npstate[2:]))
        if device.type == 'cuda' and saved.get('cuda_rng'):
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        if args.epochs <= start:
            raise ValueError('Target epochs must exceed completed epochs')
    else:
        (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    for epoch in range(start, args.epochs):
        started = time.perf_counter()
        model.train()
        total = 0
        for images, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            steps += 1
            total += loss.item() * len(images)
        result = evaluate(model, val_loader, device)
        result.update(epoch=epoch + 1, train_loss=total / len(training), seconds=round(time.perf_counter() - started, 2))
        history.append(result)
        print(json.dumps(result), flush=True)
        (args.output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        if result['auroc'] > best:
            best = result['auroc']
            atomic_save({'schema': 1, 'variant': args.variant, 'image_size': 256,
                        'labels': {'real': 0, 'fake': 1}, 'trained_steps': steps,
                        'state_dict': model.state_dict(), 'validation': result,
                        'dataset_note': dataset_note}, args.output / 'best.pt')
        npstate = np.random.get_state()
        atomic_save({'schema': 1, 'variant': args.variant, 'image_size': 256,
                     'labels': {'real': 0, 'fake': 1}, 'trained_steps': steps,
                     'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'config': config, 'data_id': data_id, 'epoch': epoch + 1,
                     'best': best, 'history': history, 'dataset_note': dataset_note,
                     'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
                     'numpy_rng': (npstate[0], npstate[1].tolist(), *npstate[2:]),
                     'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}, args.output / 'last.pt')


def atomic_save(state, path):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


if __name__ == '__main__':
    main()
