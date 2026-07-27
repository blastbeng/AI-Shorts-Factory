#!/bin/bash
set -e

echo "=== Test MVP End-to-End ==="

# Avvia il backend in background
source venv/bin/activate
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Aspetta che il server parta
sleep 5

# Crea un profilo
echo "Creazione profilo di test..."
PROFILE_RESPONSE=$(curl -s -X POST http://127.0.0.1:8000/profiles/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Profile", "genre": "Sci-Fi", "custom_prompt": "Gatti spaziali che esplorano Marte", "language": "italian", "duration_seconds": 15}')

PROFILE_ID=$(echo $PROFILE_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")
echo "Profilo creato con ID: $PROFILE_ID"

# Avvia il job
echo "Avvio job per il profilo $PROFILE_ID..."
JOB_RESPONSE=$(curl -s -X POST http://127.0.0.1:8000/jobs/$PROFILE_ID)
JOB_ID=$(echo $JOB_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")
echo "Job avviato con ID: $JOB_ID"

# Monitora il job
echo "Attendo completamento del job (max 60 secondi)..."
for i in {1..12}; do
    sleep 5
    JOB_STATUS=$(curl -s http://127.0.0.1:8000/jobs/$JOB_ID | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")
    echo "Stato job: $JOB_STATUS"
    if [ "$JOB_STATUS" == "completed" ] || [ "$JOB_STATUS" == "failed" ]; then
        break
    fi
done

# Pulizia
echo "Arresto del backend..."
kill $BACKEND_PID

if [ "$JOB_STATUS" == "completed" ]; then
    echo "=== TEST SUPERATO: Pipeline completata con successo! ==="
    exit 0
else
    echo "=== TEST FALLITO: Il job non è stato completato. ==="
    exit 1
fi
