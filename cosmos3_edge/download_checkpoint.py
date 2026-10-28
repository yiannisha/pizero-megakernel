"""Fetch pinned transformer shards and verify their upstream LFS SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

from .spec import SOURCES, verify_sources


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def download(destination):
    manifest = verify_sources()
    revision = manifest['checkpoint']['revision']
    files = json.loads((SOURCES/'weight_files.json').read_text())
    destination = Path(destination)
    for name, expected in files.items():
        path = destination/name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.stat().st_size != expected['size'] or digest(path) != expected['sha256']:
                raise ValueError(f'Existing checkpoint file failed integrity check: {path}')
            print(f'Verified existing {name}', flush=True)
            continue
        temporary = path.with_suffix(path.suffix+'.part')
        url = f'https://huggingface.co/{manifest["checkpoint"]["repo"]}/resolve/{revision}/{name}?download=true'
        h = hashlib.sha256(); count = 0
        print(f'Downloading {name} ({expected["size"]} bytes)', flush=True)
        with urllib.request.urlopen(url) as response, temporary.open('wb') as out:
            if response.status != 200:
                raise ValueError(f'Expected full checkpoint response, got {response.status}')
            for block in iter(lambda: response.read(8*1024*1024), b''):
                out.write(block); h.update(block); count += len(block)
        if count != expected['size'] or h.hexdigest() != expected['sha256']:
            raise ValueError(f'Checkpoint integrity failure: {temporary}')
        temporary.replace(path)
        print(f'Verified {name}: {h.hexdigest()}', flush=True)
    shutil.copyfile(SOURCES/'transformer__diffusion_pytorch_model.safetensors.index.json',
                    destination/'transformer/diffusion_pytorch_model.safetensors.index.json')
    shutil.copyfile(SOURCES/'transformer__config.json', destination/'transformer/config.json')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    download(args.destination)
