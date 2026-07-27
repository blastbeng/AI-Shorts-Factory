#!/bin/bash
set -e

# Carica le variabili d'ambiente dal file .env se esiste
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Uso: ./download_models.sh <nome_modello> <repo_id_huggingface> <directory_destinazione> <sezione_yaml> <chiave_yaml>
MODEL_NAME=$1
REPO_ID=$2
DEST_DIR=$3
YAML_SECTION=$4
YAML_KEY=$5
ALLOW_PATTERNS=$6

if [ -z "$MODEL_NAME" ] || [ -z "$REPO_ID" ] || [ -z "$DEST_DIR" ] || [ -z "$YAML_SECTION" ] || [ -z "$YAML_KEY" ]; then
    echo "Uso: $0 <nome_modello> <repo_id_huggingface> <directory_destinazione> <sezione_yaml> <chiave_yaml> [allow_patterns]"
    exit 1
fi

echo "Verifica del modello: $MODEL_NAME..."

# Verifica se il modello è già stato scaricato completamente
if [ -f "$DEST_DIR/.download_complete" ]; then
    echo "[OK] Modello $MODEL_NAME già presente e completo. Salto il download."
else
    echo "Modello $MODEL_NAME non trovato, incompleto o corrotto."
    
    # Se la directory esiste ma non ha il marker, cancella e ricrea per evitare conflitti
    if [ -d "$DEST_DIR" ]; then
        echo "Pulizia della directory incompleta/corrotta: $DEST_DIR"
        rm -rf "$DEST_DIR"
    fi
    
    mkdir -p "$DEST_DIR"

    PYTHON_BIN="venv/bin/python"

    if [ -f "$PYTHON_BIN" ]; then
        echo "Utilizzo di Python dall'ambiente virtuale per il download..."
        "$PYTHON_BIN" -c "
import os
from huggingface_hub import snapshot_download
allow_patterns = '$ALLOW_PATTERNS'.split() if '$ALLOW_PATTERNS' else None
token = os.getenv('HF_TOKEN')
snapshot_download(repo_id='$REPO_ID', local_dir='$DEST_DIR', local_dir_use_symlinks=False, allow_patterns=allow_patterns, token=token)
"
        
        # Crea il marker di completamento
        touch "$DEST_DIR/.download_complete"
        echo "[OK] Download completato e verificato."
    else
        echo "[ERRORE] Python non trovato in venv/bin/. Assicurati di aver eseguito scripts/install_python_environment.sh"
        exit 1
    fi
fi

echo "Aggiornamento di configs/models.yaml per $MODEL_NAME..."
PYTHON_BIN="venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi
"$PYTHON_BIN" -c "
import yaml
with open('configs/models.yaml', 'r') as f:
    config = yaml.safe_load(f)
if '$YAML_SECTION' in config and '$YAML_KEY' in config['$YAML_SECTION']:
    config['$YAML_SECTION']['$YAML_KEY']['status'] = 'installed'
    config['$YAML_SECTION']['$YAML_KEY']['path'] = '$DEST_DIR'
with open('configs/models.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
echo "Configurazione completata per $MODEL_NAME."
