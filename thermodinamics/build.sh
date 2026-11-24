#!/usr/bin/env bash
# Construye th.v para la Tang Nano 20K con N y A fijos (cableado fijo, como en el PDF)
# y lo carga en la FPGA.
#   ./build.sh               -> pregunta N y A, construye y carga en SRAM
#   ./build.sh 21 2          -> igual, sin preguntar
#   ./build.sh flash         -> pregunta N y A y graba en la flash (permanente)
#   ./build.sh 21 2 flash    -> graba en la flash sin preguntar
#   ./build.sh 21 2 build    -> solo construye
# Cada par (N, A) se construye una sola vez: build/th_N<N>_A<A>.fs
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/tools/oss-cad-suite/bin:$PATH"

# --- argumentos: los numeros son N y A, las palabras son el modo ---
N=""; A=""; MODE="sram"
for arg in "$@"; do
    case "$arg" in
        flash|build|sram) MODE="$arg" ;;
        *[!0-9]*|"") echo "argumento no reconocido: '$arg' (usa: ./build.sh [N A] [flash|build])"; exit 1 ;;
        *) if [ -z "$N" ]; then N=$arg; elif [ -z "$A" ]; then A=$arg; fi ;;
    esac
done

gcd() { local g=$1 h=$2 t; while (( h )); do t=$((g % h)); g=$h; h=$t; done; echo "$g"; }
clean() { tr -cd '0-9' <<< "$1"; }          # quita espacios, \r, etc.

# --- pide N y A hasta que sean validos ---
while true; do
    if [ -z "$N" ]; then read -rp "N = " N; N=$(clean "$N"); fi
    if [ -z "$A" ]; then read -rp "A = " A; A=$(clean "$A"); fi
    if [ -z "$N" ] || [ -z "$A" ]; then
        echo "  escribe un numero para N y otro para A"
    elif (( 10#$N < 2 || 10#$N > 48 )); then
        echo "  N=$N: N debe estar entre 2 y 48"
    elif (( $(gcd $((10#$A)) $((10#$N))) != 1 )); then
        echo "  A=$A no sirve con N=$N: gcd($A, $N) = $(gcd $((10#$A)) $((10#$N))), debe ser 1"
        echo "  (ejemplos validos: N=21 A=2, N=15 A=2, N=13 A=2, N=45 A=2)"
    else
        N=$((10#$N)); A=$((10#$A)); break
    fi
    N=""; A=""
done

mkdir -p build
B="build/th_N${N}_A${A}"
if [ -f "$B.fs" ] && [ "$B.fs" -nt th.v ] && [ "$B.fs" -nt tangnano20k.cst ]; then
    echo "$B.fs ya esta construido"
else
    echo "Sintetizando N=$N A=$A (unos 7 minutos)..."
    yosys -q -l "$B.yosys.log" -p "read_verilog th.v; chparam -set N $N -set A $A fpga_top; synth_gowin -top fpga_top -json $B.json"
    nextpnr-himbaechel -q -l "$B.nextpnr.log" --json "$B.json" --write "$B.pnr.json" \
        --device GW2AR-LV18QN88C8/I7 --vopt family=GW2A-18C --vopt cst=tangnano20k.cst \
        --freq 27
    gowin_pack -d GW2A-18C -o "$B.fs" "$B.pnr.json" 2>/dev/null
fi
grep -E "LUT4:|Max frequency" "$B.nextpnr.log" | tail -2 || true

case "$MODE" in
    build) ;;
    flash) openFPGALoader -b tangnano20k -f "$B.fs" ;;
    *)     openFPGALoader -b tangnano20k "$B.fs" ;;
esac
echo "Listo (N=$N A=$A). Monitor Serie en /dev/ttyUSB1 @115200: R, T, G, D   |   o: python3 host.py"
