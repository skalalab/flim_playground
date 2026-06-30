# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import streamlit_sortables
import os
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

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Flim-Playground',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['logo.png'],
)
