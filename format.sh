#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $0)

ruff format --config "${base_dir}/pyproject.toml" .

ruff check --config "${base_dir}/pyproject.toml" --fix .

ruff format --config "${base_dir}/pyproject.toml" .
