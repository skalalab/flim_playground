; Inno Setup script — wraps the PyInstaller onedir output (dist\Flim-Playground)
; into a single per-user installer: Flim-Playground-Setup.exe.
;
; Why per-user (PrivilegesRequired=lowest), not Program Files:
; the app saves config.toml beside its own exe via get_persistent_dir(), so the
; install directory must stay user-writable. A per-machine Program Files install
; would need admin AND would make config saves fail for standard users. In
; non-admin mode {autopf} resolves to %LOCALAPPDATA%\Programs\Flim-Playground,
; which the user owns — config writes there succeed and no UAC prompt appears.
;
; Version comes from the APP_VERSION env var (set by the CI release tag),
; falling back to a dev placeholder for manual/workflow_dispatch builds.

#define AppName "Flim-Playground"
#define AppExe "Flim-Playground.exe"
#define AppVersion GetEnv("APP_VERSION")
#if AppVersion == ""
  #define AppVersion "0.0.0-dev"
#endif

[Setup]
AppId={{8F3B2A10-9C4D-4E1F-B7A6-1D2E3F4A5B6C}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Skala Lab
DefaultDirName={autopf}\{#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=release
OutputBaseFilename=Flim-Playground-Setup
; Brands the Setup exe itself. logo.ico is generated from logo.png by the CI
; staging step (not committed) — regenerate locally with the same Pillow
; snippet in build.yml if compiling by hand.
SetupIconFile=logo.ico
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

; On an in-place upgrade, wipe the previous version's PyInstaller payload before
; copying the new one so renamed/removed bundled files (e.g. a deleted page that
; Streamlit auto-discovers) don't linger and mix versions. Scoped to _internal\
; — config.toml/analysis_config.toml live at {app}\ root beside the exe, so they
; are untouched and survive the upgrade.
[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "dist\{#AppName}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

; The uninstaller already removes everything installed via [Files]; only the
; runtime-created configs (written next to the exe, untracked by [Files]) and
; any runtime .pyc left under _internal\ need explicit cleanup. Scoped to those
; — NOT the whole {app} — so installing into a pre-existing user-chosen folder
; can't wipe unrelated files on uninstall. Runs only on uninstall, never on
; upgrade, so configs persist across upgrades.
[UninstallDelete]
Type: files; Name: "{app}\config.toml"
Type: files; Name: "{app}\analysis_config.toml"
Type: filesandordirs; Name: "{app}\_internal"
