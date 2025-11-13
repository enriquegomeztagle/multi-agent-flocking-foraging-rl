@echo off
REM Reentrenamiento para Windows (alternativa al .sh)
REM Si bash no funciona, usa este script en PowerShell/CMD

echo ============================================================================
echo REENTRENAMIENTO CON GPU - NVIDIA 4070 (Windows)
echo ============================================================================
echo.
echo Objetivo:
echo   - Hard Mode:   2M -^> 4M steps (mejorar 46%% -^> ~50%%)
echo   - Medium Mode: 2M -^> 3M steps (mejorar 72%% -^> ~76%%)
echo.
echo Tiempo estimado: ~2 horas
echo ============================================================================
echo.

REM Check CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA no disponible!'; print(f'GPU detectada: {torch.cuda.get_device_name(0)}')"
if %errorlevel% neq 0 (
    echo ERROR: GPU no detectada. Verifica instalacion de PyTorch con CUDA.
    pause
    exit /b 1
)

echo.
set /p confirm="Continuar con reentrenamiento? (y/n): "
if /i not "%confirm%"=="y" (
    echo Cancelado
    exit /b 0
)

echo.
echo ============================================================================
echo 1. REENTRENANDO HARD MODE (PRIORIDAD ALTA)
echo ============================================================================
echo.
echo Configuracion:
echo   - Config: configs/env_hard.yaml
echo   - Timesteps: 4,000,000 (2x original)
echo   - Output: models/ppo_hard/
echo   - Tiempo estimado: 40-60 minutos
echo.

REM Backup old model
if exist models\ppo_hard (
    echo Backup del modelo anterior...
    move models\ppo_hard models\ppo_hard_old_2M
)

REM Train Hard Mode
python train/train_ppo.py --config configs/env_hard.yaml --output models/ppo_hard --timesteps 4000000 --save-freq 500000
if %errorlevel% neq 0 (
    echo ERROR en entrenamiento Hard Mode
    pause
    exit /b 1
)

echo.
echo Hard Mode reentrenado!
echo.

REM Evaluate
echo Evaluando Hard Mode...
python train/eval_hard.py --model models/ppo_hard/final_model --episodes 100 --output results/hard_evaluation_4M.json
if %errorlevel% neq 0 (
    echo ERROR en evaluacion Hard Mode
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo 2. REENTRENANDO MEDIUM MODE
echo ============================================================================
echo.
echo Configuracion:
echo   - Config: configs/env_medium.yaml
echo   - Timesteps: 3,000,000 (1.5x original)
echo   - Output: models/ppo_medium/
echo   - Tiempo estimado: 30-40 minutos
echo.

REM Backup old model
if exist models\ppo_medium (
    echo Backup del modelo anterior...
    move models\ppo_medium models\ppo_medium_old_2M
)

REM Train Medium Mode
python train/train_ppo.py --config configs/env_medium.yaml --output models/ppo_medium --timesteps 3000000 --save-freq 500000
if %errorlevel% neq 0 (
    echo ERROR en entrenamiento Medium Mode
    pause
    exit /b 1
)

echo.
echo Medium Mode reentrenado!
echo.

REM Evaluate
echo Evaluando Medium Mode...
python train/eval_medium.py --model models/ppo_medium/final_model --episodes 100 --output results/medium_evaluation_3M.json
if %errorlevel% neq 0 (
    echo ERROR en evaluacion Medium Mode
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo REENTRENAMIENTO COMPLETO
echo ============================================================================
echo.
echo Resultados guardados en:
echo   - models/ppo_hard/ (4M steps)
echo   - models/ppo_medium/ (3M steps)
echo   - results/hard_evaluation_4M.json
echo   - results/medium_evaluation_3M.json
echo.
echo Modelos anteriores guardados en:
echo   - models/ppo_hard_old_2M/
echo   - models/ppo_medium_old_2M/
echo.
echo Siguiente paso: Comparar metricas
echo   python scripts/compare_retraining.py
echo.
pause
