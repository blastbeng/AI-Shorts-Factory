#!/bin/bash
set -e

# Uso: ./download_models.sh <nome_modello> <repo_id_huggingface> <directory_destinazione> <sezione_yaml> <chiave_yaml>
MODEL_NAME=$1
REPO_ID=$2
DEST_DIR=$3
YAML_SECTION=$4
YAML_KEY=$5

if [ -z "$MODEL_NAME" ] || [ -z "$REPO_ID" ] || [ -z "$DEST_DIR" ] || [ -z "$YAML_SECTION" ] || [ -z "$YAML_KEY" ]; then
    echo "Uso: $0 <nome_modello> <repo_id_huggingface> <directory_destinazione> <sezione_yaml> <chiave_yaml>"
    exit 1
fi

echo "Download del modello: $MODEL_NAME da $REPO_ID"
mkdir -p "$DEST_DIR"

if command -v huggingface-cli &> /dev/null; then
    echo "Utilizzo di huggingface-cli per il download..."
    huggingface-cli download "$REPO_ID" --local-dir "$DEST_DIR" --local-dir-use-symlinks False
else
    echo "huggingface-cli non trovato. Installa 'huggingface_hub'."
    exit 1
fi

echo "Aggiornamento di configs/models.yaml per $MODEL_NAME..."
python3 -c "
import yaml
with open('configs/models.yaml', 'r') as f:
    config = yaml.safe_load(f)
if '$YAML_SECTION' in config and '$YAML_KEY' in config['$YAML_SECTION']:
    config['$YAML_SECTION']['$YAML_KEY']['status'] = 'installed'
    config['$YAML_SECTION']['$YAML_KEY']['path'] = '$DEST_DIR'
with open('configs/models.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
echo "Download e configurazione completati per $MODEL_NAME."
