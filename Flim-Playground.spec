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

# Version stamp. ONE resolver for the whole build: src.version.get_app_version
# is the same function the running app calls, so the nav bar, the Windows .exe
# resource and the macOS Info.plist cannot disagree. In CI the APP_VERSION env
# var wins (job level in build.yml: the release tag, or 0.0.0-dev for a manual
# workflow_dispatch). Locally, with no env var, it falls through to
# `git describe` in this checkout, so a hand-built app is identifiable
# (1.11.2-4-gaeeaea1-dirty) instead of an anonymous 0.0.0-dev.
#
# The stamp goes into PyInstaller's own work dir -- already gitignored via
# `build/`, and emptied then recreated before this spec is exec'd (--clean
# included, PyInstaller/building/build_main.py:1180) -- never into the source
# tree: a build must not dirty `git status`, and a stale generated file left in
# src/ would make a developer's `streamlit run` lie about its version forever.
#
# MUST stay ABOVE Analysis(): Analysis() normalizes `datas` into a TOC at call
# time, so appending afterwards is a silent no-op -- the app would ship with no
# stamp and quietly display the fallback, with nothing in the build log.
sys.path.insert(0, SPECPATH)  # so `import src` resolves from any cwd
from src.version import STAMP_NAME, get_app_version  # noqa: E402 (SPECPATH first)

_app_version = get_app_version()
_version_stamp = os.path.join(workpath, STAMP_NAME)
with open(_version_stamp, 'w', encoding='utf-8') as _fh:
    _fh.write(_app_version + '\n')
datas.append((_version_stamp, '.'))
print(f'*** Flim-Playground version stamp: {_app_version}')

# FILEVERSION and the two macOS plist version keys are all NUMERIC formats, so
# derive the numeric prefix once, for every platform:
#   '1.11.2'                  -> (1, 11, 2, 0)
#   '1.11.2-4-gaeeaea1'       -> (1, 11, 2, 4)   # 4th field = commit count
#   '0.0.0-dev' / 'dev+abc12' -> (0, 0, 0, 0)
# The leading-digit guard matters: a tagless checkout resolves to `dev+<sha>`,
# and scraping digits out of a hex sha would invent a version like 1.0.0.
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

# onedir on all platforms: onefile re-extracted the whole ~450MB bundle to a
# temp dir on every launch (tens of seconds, worse under antivirus scanning);
# onedir materializes it once and starts in seconds. Users get
# Flim-Playground.app on macOS and a Flim-Playground/ folder on Linux (both
# tarred for download), and the same folder wrapped in a one-file installer on
# Windows — each with configs saved beside the app.

# Windows .exe version resource, built from the _app_version/_vtuple resolved
# above. Without this, PyInstaller stamps the default 0.0.0.0 into the file's
# Properties -> Details (independent of the Inno Setup installer version).
# Windows-only because the versioninfo module imports pefile, which PyInstaller
# only installs on Windows -- that is the ONLY reason for this branch, so
# nothing platform-neutral belongs inside it.
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
# macOS only (explicit no-op on other platforms): wrap the onedir folder in a
# .app bundle so Finder launches it without a Terminal window and shows the icon.
# Both plist version keys are format-constrained -- up to three
# period-separated integers -- so a raw '1.11.2-4-gaeeaea1-dirty' or
# '0.0.0-dev' is NOT legal there. The numeric prefix goes into the two
# constrained keys and the exact string into free-form CFBundleGetInfoString.
# Nothing is lost: the precise string is the nav bar's job, and the same string
# sits in Contents/Resources/VERSION.txt.
#
# Passing these off-Darwin is safe: BUNDLE.__init__ hits `if not is_darwin:
# return` before it reads any kwarg (PyInstaller/building/osx.py:49-51), and
# info_plist is merged OVER PyInstaller's defaults (osx.py:606-608), so the
# existing keys survive. Note `version=` is not decoration: it *is*
# CFBundleShortVersionString (osx.py:595) and defaults to '0.0.0' (osx.py:73),
# which is why every .app so far has reported 0.0.0 in Finder's Get Info.
_plist_version = '.'.join(str(n) for n in _vtuple[:3])  # '1.11.2' / '0.0.0'

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
