#!/usr/bin/env bash
# Build the sdist and manylinux wheels into dist/ (requires docker).
# Upload afterwards with: .build-venv/bin/twine upload dist/*
set -euo pipefail
cd "$(dirname "$0")"

git submodule update --init --recursive

python3 -m venv .build-venv
.build-venv/bin/pip install -q --upgrade build cibuildwheel twine

rm -rf dist
.build-venv/bin/python -m build --sdist --outdir dist

# Build wheels from the sdist so a missing file in the sdist fails here, not for users
.build-venv/bin/cibuildwheel --platform linux --output-dir dist dist/remixt-*.tar.gz

.build-venv/bin/twine check dist/*

if ls dist | grep -q '+'; then
    echo "WARNING: version has a local part (untagged or dirty checkout); PyPI will reject it." >&2
fi
