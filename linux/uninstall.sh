#!/usr/bin/env bash
#
# Remove the FLIM Playground application-menu launcher created by install.sh.
# This only removes the menu entry — it does NOT delete the app folder or your
# config files (config.toml / analysis_config.toml).

set -euo pipefail

DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="$DESKTOP_DIR/flim-playground.desktop"

if [ -f "$DESKTOP_FILE" ]; then
  rm -f "$DESKTOP_FILE"
  command -v update-desktop-database >/dev/null 2>&1 && \
    update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1 || true
  echo "✅ Removed the FLIM Playground menu entry."
else
  echo "No FLIM Playground menu entry found — nothing to remove."
fi
echo "   Your app folder and config files were left untouched."
