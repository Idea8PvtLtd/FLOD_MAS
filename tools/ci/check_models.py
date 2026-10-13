"""Check the shipped model files against packaging/model-checksums.sha256.

    python tools/ci/check_models.py            verify every listed file
    python tools/ci/check_models.py --tracked  also fail if git tracks a model file
                                               that the list does not name
    python tools/ci/check_models.py --update   rewrite the list from the tracked files

A model file that changes without the list changing fails the build, so a swapped or
damaged model cannot reach a release unnoticed. Run --update when a model is replaced
on purpose and commit the list together with the model.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'packaging' / 'model-checksums.sha256'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path=MANIFEST):
    entries = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            digest, name = line.split(None, 1)
            entries[name.strip()] = digest
    return entries


def tracked_models():
    out = subprocess.run(['git', 'ls-files'], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    return sorted(name for name in out.splitlines() if '/models/' in name)


def verify(root, entries):
    problems = []
    for name, digest in sorted(entries.items()):
        path = Path(root) / name
        if not path.is_file():
            problems.append(f'missing: {name}')
        elif sha256(path) != digest:
            problems.append(f'changed: {name}')
    return problems


def main(argv):
    if '--update' in argv:
        lines = ['# sha256 of every shipped model file. Written by tools/ci/check_models.py --update']
        lines += [f'{sha256(ROOT / name)}  {name}' for name in tracked_models()]
        MANIFEST.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f'wrote {MANIFEST.relative_to(ROOT)} ({len(lines) - 1} files)')
        return 0
    entries = read_manifest()
    problems = verify(ROOT, entries)
    if '--tracked' in argv:
        problems += [f'not listed: {name}' for name in tracked_models() if name not in entries]
    for problem in problems:
        print(problem)
    if problems:
        print('Model files do not match packaging/model-checksums.sha256.')
        return 1
    print(f'{len(entries)} model files match.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
