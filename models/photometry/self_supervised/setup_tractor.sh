#!/usr/bin/env bash
# Run from any directory; installs only under self_supervised/runs/.
set -euo pipefail
PHOTOMETRY_ROOT="$(cd "$(dirname "$0")" && pwd)"
TRACTOR_PYTHON="${1:-python3.11}"
TRACTOR_ENV="$PHOTOMETRY_ROOT/runs/tractor_env"
TRACTOR_DEPS="$PHOTOMETRY_ROOT/runs/tractor_dependencies"
"$TRACTOR_PYTHON" -c 'import sys; assert sys.version_info >= (3,11), "Python >=3.11 required for the pinned environment"'
"$TRACTOR_PYTHON" -m venv "$TRACTOR_ENV"
"$TRACTOR_ENV/bin/python" -m pip install --requirement "$PHOTOMETRY_ROOT/requirements-tractor-lock.txt"
export PATH="$TRACTOR_ENV/bin:$PATH"
"$TRACTOR_ENV/bin/python" -m pip install --no-build-isolation --no-deps \
  'git+https://github.com/dstndstn/tractor.git@3fd2e80eafb9cc092e203ba50a95557eb8543878'
mkdir -p "$TRACTOR_DEPS"
if [[ ! -d "$TRACTOR_DEPS/astrometry" ]]; then
  git clone https://github.com/dstndstn/astrometry.net.git "$TRACTOR_DEPS/astrometry"
fi
git -C "$TRACTOR_DEPS/astrometry" checkout d20a0503739e74b02418cde2e7d65013dadac579
"$TRACTOR_ENV/bin/python" - "$TRACTOR_DEPS" <<'PY'
import sys,sysconfig
from pathlib import Path
(Path(sysconfig.get_paths()['purelib'])/'jaisp_astrometry.pth').write_text(str(Path(sys.argv[1]).resolve())+'\n')
PY
"$TRACTOR_ENV/bin/python" -c 'from tractor import Tractor; from astrometry.util.ttime import Time; print("Tractor ready")'
