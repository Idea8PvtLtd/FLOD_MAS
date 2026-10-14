# PyInstaller recipe for the FLOD MAS inspection application (one-folder build).
#
# Build with tools/ci/build_bundle.py, not by hand: it writes BUILD_INFO.json first and
# packs and checks the result.
#
# The application finds config.json and its models next to its own modules
# (Path(__file__).parent), so the bundle keeps the source layout: every data file goes
# to the same relative folder it has in the repository.
import json
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parent
PACKAGING = ROOT / 'packaging'


def data(source, target='.'):
    return (str(ROOT / source), target)


datas = [data('config.json')]
# Every model listed in the checksum file ships, in its own folder, together with the
# list itself so the selfcheck can verify the copies inside the bundle.
for line in (PACKAGING / 'model-checksums.sha256').read_text(encoding='utf-8').splitlines():
    if line.strip() and not line.startswith('#'):
        name = line.split(None, 1)[1].strip()
        datas.append(data(name, str(Path(name).parent)))
# Untracked models present at build time (e.g. a new ribbon model) are not shipped:
# only listed, checksummed files reach a bundle.
datas.append(data('packaging/model-checksums.sha256'))
# source_model.py loads this file by path, so it must exist as a .py file on disk.
datas.append(data('inspection/local/source_reference.py', 'inspection/local'))
for optional in ('LICENSE', 'THIRD-PARTY-NOTICES.md', 'packaging/BUILD_INFO.json'):
    if (ROOT / optional).is_file():
        datas.append(data(optional))

binaries = []
hiddenimports = ['inspection.local.source_reference']
# Packages that load native code, model graphs or submodules at run time.
for package in ('mediapipe', 'onnxruntime', 'tf_keras', 'lightgbm', 'skimage', 'lazy_loader'):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception:
        continue
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

app = Analysis(
    [str(ROOT / 'main_1366.py')],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=['pytest', 'IPython', 'jupyter', 'notebook'],
    noarchive=False,
)
# Each executable carries its own module archive, so the selfcheck must be given the
# libraries and application modules it imports by name (see packaging/selfcheck.py).
import importlib.util
_spec = importlib.util.spec_from_file_location('_selfcheck', PACKAGING / 'selfcheck.py')
_selfcheck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_selfcheck)
check = Analysis(
    [str(PACKAGING / 'selfcheck.py')],
    pathex=[str(ROOT)],
    hiddenimports=list(_selfcheck.LIBRARIES + _selfcheck.APP_MODULES) + hiddenimports,
    excludes=['pytest', 'IPython', 'jupyter', 'notebook'],
    noarchive=False,
)

app_exe = EXE(
    PYZ(app.pure),
    app.scripts,
    [],
    exclude_binaries=True,
    name='FLOD_MAS',
    console=False,
    upx=False,
)
check_exe = EXE(
    PYZ(check.pure),
    check.scripts,
    [],
    exclude_binaries=True,
    name='flod-mas-selfcheck',
    console=True,
    upx=False,
)
COLLECT(
    app_exe, app.binaries, app.datas,
    check_exe, check.binaries, check.datas,
    upx=False,
    name='FLOD_MAS',
)
