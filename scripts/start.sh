#!/bin/bash
set -e

echo "=== AI Shorts Factory - Avvio Servizi ==="

# Verifica che l'ambiente virtuale esista
if [ ! -d "venv" ]; then
    echo "[ERRORE] Ambiente virtuale Python 'venv' non trovato. Esegui prima scripts/install_python_environment.sh"
    exit 1
fi

# Verifica che node_modules esista
if [ ! -d "frontend/node_modules" ]; then
    echo "[ERRORE] Dipendenze frontend non trovate. Esegui 'cd frontend && npm install'"
    exit 1
fi

# Carica le variabili d'ambiente dal file .env
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Forza PyTorch (CUDA) a vedere solo la GPU NVIDIA RTX 3060 (indice 0)
export CUDA_VISIBLE_DEVICES=0

# Ensure PyTorch's bundled NCCL and CUDA libraries are found
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SITE_PACKAGES/torch/lib:$SITE_PACKAGES/nvidia/nccl/lib:$LD_LIBRARY_PATH"

# Force loading of the correct NCCL library (required by PyTorch 2.6+)
NCCL_LIB=$(find "$SITE_PACKAGES/nvidia/nccl/lib" -name "libnccl.so*" 2>/dev/null | head -1)
if [ -n "$NCCL_LIB" ] && nm -D "$NCCL_LIB" 2>/dev/null | grep -q ncclCommResume; then
    export LD_PRELOAD="$NCCL_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "[OK] Preloading NCCL from: $NCCL_LIB"
else
    # Fallback to system NCCL if it has the symbol
    SYSTEM_NCCL=$(find /usr/lib /usr/local/lib -name "libnccl.so*" 2>/dev/null | head -1)
    if [ -n "$SYSTEM_NCCL" ] && nm -D "$SYSTEM_NCCL" 2>/dev/null | grep -q ncclCommResume; then
        export LD_PRELOAD="$SYSTEM_NCCL${LD_PRELOAD:+:$LD_PRELOAD}"
        echo "[OK] Preloading system NCCL from: $SYSTEM_NCCL"
    else
        echo "[ERRORE] Nessuna libreria NCCL con ncclCommResume trovata. PyTorch potrebbe non funzionare."
    fi
fi

# Imposta valori predefiniti se non presenti nel .env
BACKEND_HOST=${BACKEND_HOST:-"0.0.0.0"}
BACKEND_PORT=${BACKEND_PORT:-"8000"}
FRONTEND_HOST=${FRONTEND_HOST:-"0.0.0.0"}
FRONTEND_PORT=${FRONTEND_PORT:-"3000"}

# Funzione di cleanup per arrestare i processi
cleanup() {
    echo ""
    echo "=== Arresto dei servizi in corso... ==="
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID 2>/dev/null
    wait $FRONTEND_PID 2>/dev/null
    echo "Servizi arrestati correttamente."
    exit 0
}

# Intercetta il segnale di interruzione (Ctrl+C)
trap cleanup SIGINT SIGTERM

# Verify NVIDIA driver is loaded
if ! nvidia-smi &> /dev/null; then
    echo "[ERRORE] Driver NVIDIA non rilevato. Verifica che il driver sia installato e che la GPU RTX 3060 sia collegata."
    exit 1
fi

# Avvia il backend
echo "Avvio del backend (FastAPI) su ${BACKEND_HOST}:${BACKEND_PORT}..."
source venv/bin/activate
uvicorn backend.api.main:app --host $BACKEND_HOST --port $BACKEND_PORT &
BACKEND_PID=$!

# Avvia il frontend
echo "Avvio del frontend (Next.js) su ${FRONTEND_HOST}:${FRONTEND_PORT}..."
cd frontend
npm run dev -- -H $FRONTEND_HOST -p $FRONTEND_PORT &
FRONTEND_PID=$!
cd ..

echo "=================================================="
echo "Backend avviato (PID: $BACKEND_PID) -> http://${BACKEND_HOST}:${BACKEND_PORT}"
echo "Frontend avviato (PID: $FRONTEND_PID) -> http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo "Premi Ctrl+C per fermare entrambi i servizi."
echo "=================================================="

# Mantiene lo script in esecuzione e mostra i log dei processi
wait
