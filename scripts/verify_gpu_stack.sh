#!/bin/bash
set -e

echo "=== Verifica Stack GPU ==="

ERRORS=0

# Verifica CUDA (NVIDIA)
if command -v nvidia-smi &> /dev/null; then
    echo "[OK] CUDA (nvidia-smi) rilevato."
else
    echo "[AVVISO] nvidia-smi non trovato. CUDA non sembra installato."
    echo "         Installa i driver NVIDIA e CUDA toolkit manualmente se intendi usare GPU NVIDIA."
    ERRORS=$((ERRORS + 1))
fi

# Verifica ROCm (AMD)
if command -v rocm-smi &> /dev/null; then
    echo "[OK] ROCm (rocm-smi) rilevato."
else
    echo "[AVVISO] rocm-smi non trovato. ROCm non sembra installato."
    echo "         Installa i driver AMD e ROCm manualmente se intendi usare GPU AMD."
    ERRORS=$((ERRORS + 1))
fi

# Verifica Vulkan
if command -v vulkaninfo &> /dev/null; then
    echo "[OK] Vulkan (vulkaninfo) rilevato."
else
    echo "[AVVISO] vulkaninfo non trovato. Vulkan non sembra installato."
    echo "         Installa Vulkan SDK o i pacchetti vulkan-tools manualmente (necessario per llama.cpp)."
    ERRORS=$((ERRORS + 1))
fi

echo "----------------------------------------"
if [ "$ERRORS" -gt 0 ]; then
    echo "Verifica completata con $ERRORS avvisi."
    echo "NOTA: L'applicazione può ancora girare, ma alcune funzionalità AI potrebbero non essere disponibili."
else
    echo "Verifica completata: Tutti gli stack GPU sono presenti."
fi
