#!/usr/bin/env bash
# The scripts were written in a project layout with flat folders script_python/ and script_bash/, and they call each other
# (and read inputs) by paths in that layout. This helper links every file under scripts/<stage>/ into
# <root>/script_python/ (*.py) or <root>/script_bash/ (*.sh) and links module/ and config/, without changing any file.
# Usage: bash scripts/make_project_layout.sh <project root>   (the root then also holds data/ and result/)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${1:?usage: $0 <project root>}
mkdir -p "$ROOT/script_python" "$ROOT/script_bash" "$ROOT/data" "$ROOT/result"
for f in "$REPO"/scripts/[0-9]*/*.py "$REPO"/scripts/*.py; do ln -sf "$f" "$ROOT/script_python/$(basename "$f")"; done
for f in "$REPO"/scripts/[0-9]*/*.sh; do ln -sf "$f" "$ROOT/script_bash/$(basename "$f")"; done
mkdir -p "$ROOT/result/moma_serum_variant_20261029/analysis_20261030/scripts"
for f in "$REPO"/scripts/07_evaluation/lipid_uptake_sensitivity/*.py; do
  ln -sf "$f" "$ROOT/result/moma_serum_variant_20261029/analysis_20261030/scripts/$(basename "$f")"; done
[ -e "$ROOT/module" ] || ln -s "$REPO/module" "$ROOT/module"
[ -e "$ROOT/config" ] || ln -s "$REPO/config" "$ROOT/config"
# locked protocol files at the result/ paths the scripts read
(cd "$REPO/protocol" && find result -type f) | while read -r f; do
  mkdir -p "$ROOT/$(dirname "$f")"; [ -e "$ROOT/$f" ] || cp "$REPO/protocol/$f" "$ROOT/$f"; done
echo "linked into $ROOT. Set the root line at the top of each script (see README) to $ROOT."
