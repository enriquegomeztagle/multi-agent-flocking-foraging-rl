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
python3 -c "import torch; assert torch.cuda.is_available(), '❌ CUDA not available!'; print(f'✅ GPU detectada: {torch.cuda.get_device_name(0)}')"

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
PYTHONPATH=. python3 train/train_ppo.py \
    --config configs/env_hard.yaml \
    --output models/ppo_hard \
    --timesteps 4000000 \
    --save-freq 500000

echo ""
echo "✅ Hard Mode reentrenado!"
echo ""

# Evaluate immediately
echo "📊 Evaluando Hard Mode..."
PYTHONPATH=. python3 train/eval_hard.py \
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
PYTHONPATH=. python3 train/train_ppo.py \
    --config configs/env_medium.yaml \
    --output models/ppo_medium \
    --timesteps 3000000 \
    --save-freq 500000

echo ""
echo "✅ Medium Mode reentrenado!"
echo ""

# Evaluate immediately
echo "📊 Evaluando Medium Mode..."
PYTHONPATH=. python3 train/eval_medium.py \
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
echo "  python3 scripts/compare_retraining.py"
echo ""
