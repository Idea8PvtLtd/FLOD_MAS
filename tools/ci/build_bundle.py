"""Build, check and pack the application bundle for the current platform.

    python tools/ci/build_bundle.py [--version 1.2.0] [--skip-selfcheck]

Writes packaging/BUILD_INFO.json, runs PyInstaller with packaging/flod_mas.spec, runs
the bundled flod-mas-selfcheck, and packs the folder into release-assets/ as
FLOD_MAS-<version>-windows-x64.zip or FLOD_MAS-<version>-linux-x86_64.tar.gz.
Without --version the version is taken from the tag on HEAD (v1.2.0 -> 1.2.0) or,
when HEAD has no tag, 0.0.0-dev.<short commit>.
"""
import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGING = ROOT / 'packaging'
DIST = PACKAGING / 'dist'
WORK = PACKAGING / 'build'
ASSETS = ROOT / 'release-assets'


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout.strip()


def default_version():
    tag = subprocess.run(['git', 'describe', '--tags', '--exact-match', '--match', 'v*'],
                         cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return tag[1:] if tag else f'0.0.0-dev.{git("rev-parse", "--short=8", "HEAD")}'


def platform_tag():
    if sys.platform == 'win32':
        return 'windows-x64'
    if sys.platform.startswith('linux'):
        return f'linux-{platform.machine()}'
    raise SystemExit(f'unsupported build platform: {sys.platform}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version')
    parser.add_argument('--skip-selfcheck', action='store_true')
    args = parser.parse_args()
    if os.environ.get('PYTHONPATH'):
        # Packages from outside the build environment (e.g. a ROS install) would be
        # analysed and shipped. Build only from the locked environment.
        raise SystemExit('PYTHONPATH is set; unset it before building.')
    version = (args.version or default_version()).removeprefix('v')

    info = {
        'product': 'FLOD MAS',
        'version': version,
        'commit': git('rev-parse', 'HEAD'),
        'platform': platform_tag(),
        'python': platform.python_version(),
        'built_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    (PACKAGING / 'BUILD_INFO.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info))

    shutil.rmtree(DIST, ignore_errors=True)
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
                    '--distpath', str(DIST), '--workpath', str(WORK),
                    str(PACKAGING / 'flod_mas.spec')], cwd=ROOT, check=True)
    bundle = DIST / 'FLOD_MAS'

    if not args.skip_selfcheck:
        check = bundle / ('flod-mas-selfcheck.exe' if os.name == 'nt' else 'flod-mas-selfcheck')
        subprocess.run([str(check)], check=True)

    ASSETS.mkdir(exist_ok=True)
    stem = f'FLOD_MAS-{version}-{platform_tag()}'
    if os.name == 'nt':
        archive = shutil.make_archive(str(ASSETS / stem), 'zip', DIST, 'FLOD_MAS')
    else:
        archive = str(ASSETS / f'{stem}.tar.gz')
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(bundle, arcname=stem)
    print(f'packed {archive} ({Path(archive).stat().st_size / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
