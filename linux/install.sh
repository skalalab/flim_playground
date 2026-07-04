#!/usr/bin/env bash
#
# Add FLIM Playground to your application menu.
#
# The download is a self-contained folder, so there is nothing to "install" in
# the traditional sense — this just registers a clickable, iconed launcher
# (a freedesktop .desktop entry) pointing at the app where it currently sits, so
# you can start it from your app menu / Activities like a native program instead
# of running the binary by hand. Re-run this if you move the folder. Undo any
# time with ./uninstall.sh — neither script touches your data.

set -euo pipefail

APP_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$APP_DIR/Flim-Playground"
ICON="$APP_DIR/flim-playground.png"
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="$DESKTOP_DIR/flim-playground.desktop"

if [ ! -f "$BIN" ]; then
  echo "Error: could not find the Flim-Playground launcher next to this script ($BIN)." >&2
  echo "Run install.sh from inside the extracted Flim-Playground folder." >&2
  exit 1
fi
chmod +x "$BIN" 2>/dev/null || true   # restore the exec bit if the download lost it

mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=FLIM Playground
Comment=Interactive single-cell FLIM data extraction and analysis
Exec="$BIN"
Icon=$ICON
Terminal=false
Categories=Education;Science;
StartupNotify=true
EOF

# Refresh the menu database if the helper exists (best-effort; not on every distro).
command -v update-desktop-database >/dev/null 2>&1 && \
  update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1 || true

echo "✅ FLIM Playground is now in your application menu."
echo "   Search for \"FLIM Playground\" in your launcher / Activities and click it to run."
echo "   Moved the folder? Just run ./install.sh again. Remove it with ./uninstall.sh."
