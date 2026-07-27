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
    export $(grep -v '^#' .env | xargs)
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
