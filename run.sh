#!/usr/bin/env bash
# Run TCG Card Centering Detector from source
set -euo pipefail
cd "$(dirname "$0")"
python3 -m card_centering "$@"
