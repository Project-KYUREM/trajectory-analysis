#!/usr/bin/env bash
# Download the SPICE kernels kyurem needs into data/. They are not committed
# (de440.bsp alone is 115 MB, over GitHub's 100 MB limit).
set -euo pipefail
cd "$(dirname "$0")/../data" 2>/dev/null || { mkdir -p "$(dirname "$0")/../data"; cd "$(dirname "$0")/../data"; }

NAIF=https://naif.jpl.nasa.gov/pub/naif/generic_kernels
fetch () { [ -f "$2" ] && echo "have $2" || { echo "get $2"; curl -sSL --fail "$1" -o "$2"; }; }

fetch "$NAIF/spk/planets/de440.bsp"                de440.bsp
fetch "$NAIF/lsk/naif0012.tls"                     naif0012.tls
fetch "$NAIF/pck/pck00011.tpc"                     pck00011.tpc
fetch "$NAIF/pck/gm_de440.tpc"                     gm_de440.tpc
fetch "$NAIF/spk/satellites/nep097.bsp"            nep097.bsp

# Neptune-GRAM mean atmosphere profile, as tabulated by AMAT (Girija et al.).
# AMAT is GPL-3.0, so the table is downloaded rather than committed to this
# MIT repo. Used by aerocapture/atmosphere.py.
AMAT=https://raw.githubusercontent.com/athulpg007/AMAT/master
fetch "$AMAT/atmdata/Neptune/neptune-gram-avg.dat" neptune-gram-avg.dat
echo "kernels ready in $(pwd)"
