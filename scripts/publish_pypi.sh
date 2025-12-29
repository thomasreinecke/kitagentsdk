# scripts/publish_pypi.sh
set -euo pipefail

info() { echo "ℹ️  $*"; }
warn() { echo "⚠️  $*" >&2; }
fail() { echo "❌ $*" >&2; exit 1; }

if [ ! -f "pyproject.toml" ]; then
  fail "pyproject.toml not found. Run this from the kitagentsdk repo root."
fi

if [ -z "${TWINE_USERNAME:-}" ]; then
  cat >&2 <<'EOF'
❌ TWINE_USERNAME is not set.

If you're using a PyPI API token, set:
  export TWINE_USERNAME="__token__"
EOF
  exit 1
fi

if [ -z "${TWINE_PASSWORD:-}" ]; then
  cat >&2 <<'EOF'
❌ TWINE_PASSWORD is not set.

Set it to your PyPI API token (starts with "pypi-..."):
  export TWINE_PASSWORD="pypi-REPLACE_WITH_YOUR_TOKEN"
EOF
  exit 1
fi

if [ "${TWINE_USERNAME}" != "__token__" ]; then
  warn "TWINE_USERNAME is '${TWINE_USERNAME}'. If you're using an API token, it should be '__token__'."
fi

case "${TWINE_PASSWORD}" in
  pypi-*) ;;
  *)
    warn "TWINE_PASSWORD does not start with 'pypi-'. If you're using an API token, double-check it's the token value."
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  fail "Python executable '${PYTHON_BIN}' not found. Set PYTHON_BIN or ensure python is on PATH."
fi

if ! "${PYTHON_BIN}" -m twine --version >/dev/null 2>&1; then
  cat >&2 <<EOF
❌ 'twine' is not available in: ${PYTHON_BIN}

Install it (recommended in a venv):
  ${PYTHON_BIN} -m pip install --upgrade twine
EOF
  exit 1
fi

if ! "${PYTHON_BIN}" -m build --version >/dev/null 2>&1; then
  cat >&2 <<EOF
❌ 'build' is not available in: ${PYTHON_BIN}

Install it:
  ${PYTHON_BIN} -m pip install --upgrade build
EOF
  exit 1
fi

info "Using Python: $(${PYTHON_BIN} --version 2>&1)"
info "twine: $(${PYTHON_BIN} -m twine --version 2>&1 | head -n 1)"
info "Cleaning dist/ ..."
rm -rf dist build *.egg-info 2>/dev/null || true

info "Building sdist + wheel ..."
"${PYTHON_BIN}" -m build

info "Checking distributions ..."
"${PYTHON_BIN}" -m twine check dist/*

info "Uploading to PyPI ..."
set +e
UPLOAD_OUT="$("${PYTHON_BIN}" -m twine upload dist/* 2>&1)"
UPLOAD_RC=$?
set -e

if [ ${UPLOAD_RC} -ne 0 ]; then
  echo "${UPLOAD_OUT}" >&2
  cat >&2 <<'EOF'

❌ Upload failed.

Common causes:
- Version already exists on PyPI → bump [project].version in pyproject.toml and rebuild.
- Token lacks permission → verify token scope for this project.
- You’re uploading to the wrong index → for TestPyPI use: twine upload -r testpypi dist/*
EOF
  exit ${UPLOAD_RC}
fi

echo "${UPLOAD_OUT}"
info "Done."
