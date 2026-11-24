"""Download author-published CIFAKE; deterministic, disjoint, balanced subset."""
import csv
import hashlib
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = 'jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images'


def download(url, target):
    request = urllib.request.Request(url, headers={'User-Agent': 'FakeCatcher-Research'})
    with urllib.request.urlopen(request, timeout=120) as response, target.with_suffix('.part').open('wb') as out:
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > 600 * 1024 * 1024:
                raise ValueError('Archive exceeds 600MB budget')
            out.write(chunk)
    target.with_suffix('.part').replace(target)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-per-class', type=int, default=4000)
    parser.add_argument('--val-per-class', type=int, default=500)
    parser.add_argument('--test-per-class', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'cifake_pilot')
    args = parser.parse_args()
    if min(args.train_per_class, args.val_per_class, args.test_per_class) < 1:
        parser.error('Counts must be positive')
    if args.output.exists():
        raise FileExistsError(f'Will not overwrite {args.output}')
    cache = ROOT / 'data' / 'downloads'
    cache.mkdir(parents=True, exist_ok=True)
    revision_file = cache / 'revision.json'
    if not revision_file.exists():
        download(f'https://api.github.com/repos/{REPO}/commits/main', revision_file)
    revision = json.loads(revision_file.read_text(encoding='utf-8'))['sha']
    archive = cache / f'cifake-{revision}.zip'
    if not archive.exists():
        print('Downloading CIFAKE author archive...', flush=True)
        download(f'https://codeload.github.com/{REPO}/zip/{revision}', archive)
    args.output.mkdir(parents=True)
    seen, rows = set(), []
    rng = random.Random(args.seed)
    with zipfile.ZipFile(archive) as z:
        for source_split in ['train', 'test']:
            for label in ['real', 'fake']:
                names = sorted(n for n in z.namelist() if f'/dataset/{source_split}/{label}/' in n.lower() and n.lower().endswith(('.jpg', '.jpeg', '.png')))
                rng.shuffle(names)
                targets = [('train', args.train_per_class), ('val', args.val_per_class)] if source_split == 'train' else [('test', args.test_per_class)]
                iterator = iter(names)
                for split, count in targets:
                    folder = args.output / split / label
                    folder.mkdir(parents=True)
                    kept = 0
                    for name in iterator:
                        with Image.open(io.BytesIO(z.read(name))) as im:
                            image = im.convert('RGB')
                        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                        if digest in seen:
                            continue
                        seen.add(digest)
                        relative = f'{split}/{label}/{digest}.png'
                        image.save(args.output / relative)
                        rows.append({'path': relative, 'label': label, 'split': split,
                                     'source': name, 'generator': 'SD1.4' if label == 'fake' else 'CIFAR10',
                                     'pixel_sha256': digest, 'width': image.width, 'height': image.height})
                        kept += 1
                        if kept == count:
                            break
                    if kept != count:
                        raise ValueError(f'Not enough unique images: {split}/{label} {kept}/{count}')
                    print(f'{split}/{label}: {kept}', flush=True)
        readme = next(n for n in z.namelist() if n.count('/') == 1 and n.endswith('README.md'))
        (args.output / 'SOURCE_AND_LICENSE.md').write_bytes(z.read(readme))
    with (args.output / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    info = {'source': f'https://github.com/{REPO}', 'revision': revision, 'seed': args.seed,
            'archive_sha256': hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest(),
            'counts': {s: sum(r['split'] == s for r in rows) for s in ['train', 'val', 'test']},
            'note': 'CIFAKE 32x32 subset pilot; not validated on unseen generators or high-resolution uploads.'}
    (args.output / 'dataset.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
