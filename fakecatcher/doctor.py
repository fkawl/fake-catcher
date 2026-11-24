import json
import platform
import shutil
import sys
from pathlib import Path
import torch
import torchvision

def main():
    result = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
              'torchvision': torchvision.__version__, 'cuda_runtime': torch.version.cuda,
              'cuda_available': torch.cuda.is_available(),
              'free_disk_gb': round(shutil.disk_usage(Path.cwd()).free / 1024**3, 1)}
    if torch.cuda.is_available():
        result['gpu'] = torch.cuda.get_device_name(0)
        result['vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        result['gpu_compute_ok'] = bool(torch.isfinite(torch.rand(16, 16, device='cuda') @ torch.rand(16, 16, device='cuda')).all())
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
