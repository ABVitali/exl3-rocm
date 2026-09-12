#!/usr/bin/env bash
# Clone the three upstream trees this overlay builds against, at the commits it was developed on.
set -euo pipefail
cd "$(dirname "$0")"
clone() { [ -d "$2" ] || git clone -q "$1" "$2"; git -C "$2" checkout -q "$3"; echo "$2 @ $3"; }
clone https://github.com/turboderp-org/exllamav3.git upstream-exllamav3 be57335b087e
clone https://github.com/vcruz305/vllm-exl3.git      upstream-vllm-exl3 8f4517e80416
clone https://github.com/Zeuss5/cuda-exl3.git        upstream-cuda-exl3 6a1ffc34866e
