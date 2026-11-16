#!/usr/bin/env bash
# ==============================================================================
# Instalador de entorno para TokenSafe Remaster (Node.js + @huggingface/transformers)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo " Configurando TokenSafe Remaster (Qwen3-Embedding-0.6B-ONNX + JS)"
echo "=================================================================="

# 1. Instalar Node.js y npm si no están disponibles
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    echo "[1/2] Instalando Node.js y npm..."
    sudo apt-get update -qq
    sudo apt-get install -y nodejs npm
else
    echo "[1/2] ✔ Node.js ($(node --version)) y npm ($(npm --version)) ya están instalados."
fi

# 2. Instalar @huggingface/transformers localmente
echo "[2/2] Instalando dependencias en node_modules (@huggingface/transformers)..."
npm install --no-audit --no-fund

echo ""
echo "=================================================================="
echo " ✅ TokenSafe Remaster listo."
echo "=================================================================="
echo " 1. Ejecutar tests unitarios y de integración (System One):"
echo "    cd $SCRIPT_DIR && node bench/test_remaster.mjs"
echo ""
echo " 2. Ejecutar benchmark de las 8 suites en consola:"
echo "    cd $SCRIPT_DIR && node bench/eval_accuracy.mjs --limit 5"
echo ""
echo " 3. Abrir el Playground Web + Benchmarks WebGPU/WASM:"
echo "    cd $SCRIPT_DIR && python3 -m http.server 8080"
echo "    (Abre http://localhost:8080 y http://localhost:8080/bench.html)"
echo "=================================================================="
