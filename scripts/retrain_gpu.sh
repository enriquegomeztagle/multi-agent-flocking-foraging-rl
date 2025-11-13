#!/bin/bash
# Reentrenamiento optimizado para GPU NVIDIA 4070
# Mejora Hard Mode (prioridad) y Medium Mode

set -e  # Exit on error

echo "=============================================================================="
echo "🚀 REENTRENAMIENTO CON GPU - NVIDIA 4070"
echo "=============================================================================="
echo ""
echo "Objetivo:"
echo "  • Hard Mode:   2M → 4M steps (mejorar 46% → ~50%)"
echo "  • Medium Mode: 2M → 3M steps (mejorar 72% → ~76%)"
echo ""
echo "Tiempo estimado: ~2 horas"
echo "=============================================================================="
echo ""

# Check if CUDA is available
if ! python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "❌ CUDA not available!"
    echo ""
    echo "Diagnóstico:"
    python -c "import torch; print(f'  • PyTorch version: {torch.__version__}'); print(f'  • CUDA available: {torch.cuda.is_available()}'); print(f'  • CUDA version: {torch.version.cuda if torch.version.cuda else \"N/A\"}')" 2>&1 || echo "  • Error checking PyTorch"
    echo ""
    echo "Solución:"
    echo "  Tienes PyTorch CPU-only instalado. Para usar GPU, instala PyTorch con CUDA:"
    echo "  pip uninstall torch torchvision torchaudio"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    echo ""
    read -p "¿Continuar con CPU (MUCHO más lento)? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Cancelado"
        exit 1
    fi
    echo "⚠️  Usando CPU - El entrenamiento será muy lento"
else
    python -c "import torch; print(f'✅ GPU detectada: {torch.cuda.get_device_name(0)}')"
fi

echo ""
read -p "¿Continuar con reentrenamiento? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelado"
    exit 1
fi

echo ""
echo "=============================================================================="
echo "1️⃣  REENTRENANDO HARD MODE (PRIORIDAD ALTA)"
echo "=============================================================================="
echo ""
echo "Configuración:"
echo "  • Config: configs/env_hard.yaml"
echo "  • Timesteps: 4,000,000 (2x original)"
echo "  • Output: models/ppo_hard_retrained/"
echo "  • Tiempo estimado: 40-60 minutos"
echo ""

# Backup old model
if [ -d "models/ppo_hard" ]; then
    echo "📦 Backup del modelo anterior..."
    mv models/ppo_hard models/ppo_hard_old_2M
fi

# Train Hard Mode
PYTHONPATH=. python train/train_ppo.py \
    --config configs/env_hard.yaml \
    --output models/ppo_hard \
    --timesteps 4000000 \
    --save-freq 500000

echo ""
echo "✅ Hard Mode reentrenado!"
echo ""

# Evaluate immediately
echo "📊 Evaluando Hard Mode..."
PYTHONPATH=. python train/eval_hard.py \
    --model models/ppo_hard/final_model \
    --episodes 100 \
    --output results/hard_evaluation_4M.json

echo ""
echo "=============================================================================="
echo "2️⃣  REENTRENANDO MEDIUM MODE"
echo "=============================================================================="
echo ""
echo "Configuración:"
echo "  • Config: configs/env_medium.yaml"
echo "  • Timesteps: 3,000,000 (1.5x original)"
echo "  • Output: models/ppo_medium_retrained/"
echo "  • Tiempo estimado: 30-40 minutos"
echo ""

# Backup old model
if [ -d "models/ppo_medium" ]; then
    echo "📦 Backup del modelo anterior..."
    mv models/ppo_medium models/ppo_medium_old_2M
fi

# Train Medium Mode
PYTHONPATH=. python train/train_ppo.py \
    --config configs/env_medium.yaml \
    --output models/ppo_medium \
    --timesteps 3000000 \
    --save-freq 500000

echo ""
echo "✅ Medium Mode reentrenado!"
echo ""

# Evaluate immediately
echo "📊 Evaluando Medium Mode..."
PYTHONPATH=. python train/eval_medium.py \
    --model models/ppo_medium/final_model \
    --episodes 100 \
    --output results/medium_evaluation_3M.json

echo ""
echo "=============================================================================="
echo "✅ REENTRENAMIENTO COMPLETO"
echo "=============================================================================="
echo ""
echo "Resultados guardados en:"
echo "  • models/ppo_hard/ (4M steps)"
echo "  • models/ppo_medium/ (3M steps)"
echo "  • results/hard_evaluation_4M.json"
echo "  • results/medium_evaluation_3M.json"
echo ""
echo "Modelos anteriores guardados en:"
echo "  • models/ppo_hard_old_2M/"
echo "  • models/ppo_medium_old_2M/"
echo ""
echo "Siguiente paso: Comparar métricas"
echo "  python scripts/compare_retraining.py"
echo ""
