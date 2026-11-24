"""One command: matched RGB / RGB+FFT pilot training and held-out evaluation."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/cifake_pilot'))
    parser.add_argument('--runs', type=Path, default=Path('runs/cifake_pilot'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    for variant in ['rgb', 'rgb_fft']:
        output = args.runs / variant
        base = [sys.executable, '-m', 'fakecatcher.train', '--data', str(args.data),
                '--variant', variant, '--batch-size', str(args.batch_size), '--device', args.device]
        checkpoint = output / 'last.pt'
        finished = 0
        if checkpoint.exists():
            import torch
            finished = torch.load(checkpoint, map_location='cpu', weights_only=True)['epoch']
        if finished < args.epochs:
            cmd = base + ['--output', str(output), '--epochs', str(args.epochs)]
            if checkpoint.exists():
                cmd += ['--resume', str(checkpoint)]
            subprocess.run(cmd, check=True)
        evaluation = output / f'evaluation_epoch_{args.epochs}'
        if not (evaluation / 'test.json').exists():
            subprocess.run(base + ['--output', str(evaluation), '--evaluate', str(output / 'best.pt')], check=True)
        print(variant, json.loads((evaluation / 'test.json').read_text()), flush=True)


if __name__ == '__main__':
    main()
