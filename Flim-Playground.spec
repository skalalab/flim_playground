# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import streamlit_sortables
import os
import sys
import streamlit_theme

datas = [('src', 'src'), ('pages', 'pages'), ('main.py', '.'), ('launcher.py', '.'), ('logo.png', '.'), ('.streamlit', '.streamlit')]
binaries = []
hiddenimports = ['pages.data_analysis', 'pages.data_extraction', 'psutil']

# Add streamlit
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Add imblearn
tmp_ret = collect_all('imblearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Add streamlit-sortables with explicit frontend build directory
tmp_ret = collect_all('streamlit-sortables')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Manually add streamlit-sortables frontend build directory
sortables_path = os.path.dirname(streamlit_sortables.__file__)
frontend_build_path = os.path.join(sortables_path, 'frontend', 'build')
if os.path.exists(frontend_build_path):
    datas.append((frontend_build_path, 'streamlit_sortables/frontend/build'))

# Add streamlit-theme
tmp_ret = collect_all('streamlit_theme')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Manually add streamlit-theme frontend build directory
# st-theme uses 'frontend/dist' unlike sortables which uses 'frontend/build'
st_theme_path = os.path.dirname(streamlit_theme.__file__)
st_theme_frontend_path = os.path.join(st_theme_path, 'frontend', 'dist')
if os.path.exists(st_theme_frontend_path):
    datas.append((st_theme_frontend_path, 'streamlit_theme/frontend/dist'))

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

# onedir on all platforms: onefile re-extracted the whole ~450MB bundle to a
# temp dir on every launch (tens of seconds, worse under antivirus scanning);
# onedir materializes it once and starts in seconds. Users get
# Flim-Playground.app on macOS and a Flim-Playground/ folder on Linux (both
# tarred for download), and the same folder wrapped in a one-file installer on
# Windows — each with configs saved beside the app.

# Windows .exe version resource. Without this, PyInstaller stamps the default
# 0.0.0.0 into the file's Properties -> Details (independent of the Inno Setup
# installer version). Build a VSVersionInfo from APP_VERSION (set at job level
# in build.yml: the release tag for release builds, else a dev placeholder) so
# the .exe reports the same version as the installer. Windows-only: the
# versioninfo module imports pefile, which PyInstaller only installs on Windows.
version_resource = None
if sys.platform == 'win32':
    import re
    from PyInstaller.utils.win32.versioninfo import (
        VSVersionInfo, FixedFileInfo, StringFileInfo, StringTable, StringStruct,
        VarFileInfo, VarStruct,
    )
    _app_version = os.environ.get('APP_VERSION') or '0.0.0-dev'
    # FILEVERSION/PRODUCTVERSION are 4 ints; take the leading numeric parts
    # ('1.11.0' -> (1, 11, 0, 0); '0.0.0-dev' -> (0, 0, 0, 0)).
    _nums = [int(n) for n in re.findall(r'\d+', _app_version)][:4]
    _nums += [0] * (4 - len(_nums))
    _vtuple = tuple(_nums)
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
# macOS only (explicit no-op on other platforms): wrap the onedir folder in a
# .app bundle so Finder launches it without a Terminal window and shows the icon.
app = BUNDLE(
    coll,
    name='Flim-Playground.app',
    icon='logo.png',
    bundle_identifier='com.skalalab.flim-playground',
)
