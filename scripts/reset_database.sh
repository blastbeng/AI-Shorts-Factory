#!/bin/bash
set -e

echo "=== AI Shorts Factory - Reset Database ==="

# Verifica che il backend non sia in esecuzione (controllo semplice)
if pgrep -f "uvicorn backend.api.main:app" > /dev/null; then
    echo "[ERRORE] Il backend sembra essere in esecuzione. Arrestalo prima di eseguire questo script."
    exit 1
fi

DB_FILE="aishorts.db"

if [ -f "$DB_FILE" ]; then
    echo "Rimozione del file database esistente ($DB_FILE)..."
    rm -f "$DB_FILE"
    echo "[OK] Database eliminato. Verrà ricreato automaticamente con lo schema aggiornato al prossimo avvio del backend."
else
    echo "[AVVISO] Nessun file $DB_FILE trovato. Nessuna azione necessaria."
fi
