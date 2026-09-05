# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import streamlit_sortables
import os
import re
import sys

datas = [('src', 'src'), ('pages', 'pages'), ('main.py', '.'), ('launcher.py', '.'), ('logo.png', '.'), ('.streamlit', '.streamlit')]
binaries = []
hiddenimports = ['pages.data_analysis', 'pages.data_extraction', 'psutil']

# Add streamlit
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Add imblearn
tmp_ret = collect_all('imblearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Add python-calamine — the spreadsheet reader. pandas pulls it in lazily through
# import_optional_dependency, so static analysis never sees the import and the
# compiled extension module has to be collected explicitly.
tmp_ret = collect_all('python_calamine')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
hiddenimports += ['pandas.io.excel._calamine']

# Add streamlit-sortables with explicit frontend build directory
tmp_ret = collect_all('streamlit-sortables')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Manually add streamlit-sortables frontend build directory
sortables_path = os.path.dirname(streamlit_sortables.__file__)
frontend_build_path = os.path.join(sortables_path, 'frontend', 'build')
if os.path.exists(frontend_build_path):
    datas.append((frontend_build_path, 'streamlit_sortables/frontend/build'))

# Use the runtime version resolver for the app stamp and platform metadata.
# APP_VERSION takes precedence in CI; local builds fall back to git describe.
# Write the stamp in PyInstaller's work directory to keep generated data out of
# the source tree. Add it before Analysis(), which snapshots datas when called.
sys.path.insert(0, SPECPATH)  # so `import src` resolves from any cwd
from src.version import STAMP_NAME, get_app_version  # noqa: E402 (SPECPATH first)

_app_version = get_app_version()
_version_stamp = os.path.join(workpath, STAMP_NAME)
with open(_version_stamp, 'w', encoding='utf-8') as _fh:
    _fh.write(_app_version + '\n')
datas.append((_version_stamp, '.'))
print(f'*** Flim-Playground version stamp: {_app_version}')

# Platform version fields require numbers. Include the commit count as the fourth
# field when available; ignore tagless dev+<sha> strings to avoid parsing hash digits.
_nums = [int(n) for n in re.findall(r'\d+', _app_version)][:4] if _app_version[:1].isdigit() else []
_nums += [0] * (4 - len(_nums))
_vtuple = tuple(_nums)

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# Use onedir to avoid extracting the bundle on each launch. Distribution wraps it
# as a macOS app, a Linux folder, or a Windows installer.

# Windows file metadata uses the same resolved version. Keep versioninfo imports
# inside this branch because their pefile dependency is Windows-only.
version_resource = None
if sys.platform == 'win32':
    from PyInstaller.utils.win32.versioninfo import (
        VSVersionInfo, FixedFileInfo, StringFileInfo, StringTable, StringStruct,
        VarFileInfo, VarStruct,
    )
    version_resource = VSVersionInfo(
        ffi=FixedFileInfo(
            filevers=_vtuple, prodvers=_vtuple,
            mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0,
            date=(0, 0),
        ),
        kids=[
            StringFileInfo([
                StringTable('040904B0', [
                    StringStruct('CompanyName', 'Skala Lab'),
                    StringStruct('FileDescription', 'FLIM Playground'),
                    StringStruct('FileVersion', _app_version),
                    StringStruct('InternalName', 'Flim-Playground'),
                    StringStruct('OriginalFilename', 'Flim-Playground.exe'),
                    StringStruct('ProductName', 'FLIM Playground'),
                    StringStruct('ProductVersion', _app_version),
                ]),
            ]),
            VarFileInfo([VarStruct('Translation', [1033, 1200])]),
        ],
    )

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Flim-Playground',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['logo.png'],
    version=version_resource,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Flim-Playground',
)
# BUNDLE wraps the folder as a macOS app and is a no-op on other platforms.
# Its constrained version keys use three numeric fields; the full version remains
# in CFBundleGetInfoString and VERSION.txt.
_plist_version = '.'.join(str(n) for n in _vtuple[:3])

app = BUNDLE(
    coll,
    name='Flim-Playground.app',
    icon='logo.png',
    bundle_identifier='com.skalalab.flim-playground',
    version=_plist_version,  # -> CFBundleShortVersionString
    info_plist={
        'CFBundleVersion': _plist_version,
        'CFBundleGetInfoString': f'FLIM Playground {_app_version}',
    },
)
