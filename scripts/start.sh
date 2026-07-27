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
echo "Avvio del backend (FastAPI) sulla porta 8000..."
source venv/bin/activate
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Avvia il frontend
echo "Avvio del frontend (Next.js) sulla porta 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "=================================================="
echo "Backend avviato (PID: $BACKEND_PID) -> http://localhost:8000"
echo "Frontend avviato (PID: $FRONTEND_PID) -> http://localhost:3000"
echo "Premi Ctrl+C per fermare entrambi i servizi."
echo "=================================================="

# Mantiene lo script in esecuzione e mostra i log dei processi
wait
