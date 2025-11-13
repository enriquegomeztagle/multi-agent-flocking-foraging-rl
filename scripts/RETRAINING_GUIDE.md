# 🚀 Guía de Reentrenamiento con GPU

Instrucciones para reentrenar Hard Mode y Medium Mode con más timesteps para mejorar métricas.

---

## ⚙️ Requisitos

- **GPU:** NVIDIA 4070 (o superior)
- **CUDA:** Instalado y funcionando
- **Tiempo:** ~2 horas para ambos modos
- **Espacio:** ~100 MB adicional

---

## 🎯 Objetivos del Reentrenamiento

### Hard Mode (PRIORIDAD ALTA)
- **Actual:** 45.90% eficiencia (2M steps)
- **Objetivo:** 48-52% eficiencia (4M steps)
- **Razón:** Demasiado cerca del límite inferior (40%)

### Medium Mode (PRIORIDAD MEDIA)
- **Actual:** 72.55% eficiencia (2M steps)
- **Objetivo:** 75-78% eficiencia (3M steps)
- **Razón:** Alta varianza (std=288), podría ser más consistente

---

## 🚀 Opción 1: Reentrenar Ambos (Recomendado)

Ejecuta el script completo que entrena y evalúa ambos modos:

```bash
# Hacer ejecutable
chmod +x scripts/retrain_gpu.sh

# Ejecutar reentrenamiento completo
bash scripts/retrain_gpu.sh
```

**Esto hará:**
1. Backup de modelos actuales (→ `_old_2M`)
2. Reentrenar Hard Mode con 4M steps (~40-60 min)
3. Evaluar Hard Mode inmediatamente
4. Reentrenar Medium Mode con 3M steps (~30-40 min)
5. Evaluar Medium Mode inmediatamente

---

## 🎯 Opción 2: Reentrenar Solo Hard Mode

Si solo quieres mejorar Hard Mode (lo más crítico):

```bash
# Backup del modelo actual
mv models/ppo_hard models/ppo_hard_old_2M

# Reentrenar con 4M steps
PYTHONPATH=. python3 train/train_ppo.py \
    --config configs/env_hard.yaml \
    --output models/ppo_hard \
    --timesteps 4000000 \
    --save-freq 500000

# Evaluar
PYTHONPATH=. python3 train/eval_hard.py \
    --model models/ppo_hard/final_model \
    --episodes 100 \
    --output results/hard_evaluation_4M.json
```

---

## 🎯 Opción 3: Reentrenar Solo Medium Mode

Si solo quieres mejorar Medium Mode:

```bash
# Backup del modelo actual
mv models/ppo_medium models/ppo_medium_old_2M

# Reentrenar con 3M steps
PYTHONPATH=. python3 train/train_ppo.py \
    --config configs/env_medium.yaml \
    --output models/ppo_medium \
    --timesteps 3000000 \
    --save-freq 500000

# Evaluar
PYTHONPATH=. python3 train/eval_medium.py \
    --model models/ppo_medium/final_model \
    --episodes 100 \
    --output results/medium_evaluation_3M.json
```

---

## 📊 Comparar Resultados

Después del reentrenamiento, compara los resultados:

```bash
python3 scripts/compare_retraining.py
```

Esto mostrará:
- Eficiencia antes vs después
- Reducción de varianza (std)
- Mejora en Gini (equidad)
- Cambio en distribución de tiers

---

## 🔧 Ajustes para GPU 4070

Los scripts ya están optimizados para tu GPU. PyTorch detectará automáticamente CUDA.

**Verificar GPU:**
```bash
python3 -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Esperado:**
```
CUDA disponible: True
GPU: NVIDIA GeForce RTX 4070
```

---

## 📈 Métricas de Éxito

### Hard Mode
- ✅ Eficiencia: 48-52% (centro del target 40-50%)
- ✅ Std < 230 (reducir inconsistencia)
- ✅ Gini < 0.50 (mejorar equidad)
- ✅ Episodios <40%: < 25 (reducir de 34)

### Medium Mode
- ✅ Eficiencia: 75-78% (bien sobre target 70%)
- ✅ Std < 250 (reducir de 288)
- ✅ Episodios ≥70%: > 65 (aumentar de 52)

---

## 🗂️ Estructura de Archivos

**Modelos reentrenados:**
```
models/
├── ppo_hard/               # Nuevo (4M steps)
│   ├── final_model.zip
│   ├── vecnormalize.pkl
│   └── env_config.yaml
├── ppo_hard_old_2M/        # Backup del anterior
├── ppo_medium/             # Nuevo (3M steps)
└── ppo_medium_old_2M/      # Backup del anterior
```

**Resultados:**
```
results/
├── hard_evaluation.json        # Original (2M)
├── hard_evaluation_4M.json     # Nuevo (4M)
├── medium_evaluation.json      # Original (2M)
└── medium_evaluation_3M.json   # Nuevo (3M)
```

---

## ⏱️ Tiempos Estimados (GPU 4070)

| Modo | Steps | Tiempo GPU | Evaluación | Total |
|------|-------|------------|------------|-------|
| Hard | 4M | 40-60 min | 3-5 min | ~1 hora |
| Medium | 3M | 30-40 min | 3-5 min | ~45 min |
| **TOTAL** | | **~1.5h** | **~10 min** | **~2 horas** |

---

## 🚨 Troubleshooting

### Error: CUDA out of memory
Si la GPU se queda sin memoria:
```bash
# Reduce parallel environments (edita train_ppo.py)
# Línea ~50: n_envs=4 → n_envs=2
```

### Error: CUDA not available
Verifica instalación de PyTorch con CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Los resultados no mejoran
Si después de 4M steps Hard Mode sigue <48%:
- Considera 5M steps
- Revisa logs de TensorBoard (si están habilitados)
- Verifica que no hay overfitting

---

## 📝 Notas Finales

- Los modelos `_old_2M` son backups - NO los borres hasta verificar que los nuevos son mejores
- Puedes restaurar un backup con: `mv models/ppo_hard_old_2M models/ppo_hard`
- El reentrenamiento NO modifica `episode_len` - eso es parte del diseño
- Solo aumenta los `training timesteps` para mejor convergencia

---

## ✅ Checklist de Reentrenamiento

- [ ] Verificar GPU disponible: `nvidia-smi`
- [ ] Hacer backup de modelos actuales
- [ ] Ejecutar reentrenamiento (Opción 1, 2 o 3)
- [ ] Evaluar modelos nuevos (100 episodios)
- [ ] Comparar resultados: `python3 scripts/compare_retraining.py`
- [ ] Si mejora ≥3pp: Actualizar documentación (HARD_MODE.md, MEDIUM_MODE.md, README.md)
- [ ] Si mejora <2pp: Considerar más timesteps o ajustes

---

**¿Listo para empezar?**

```bash
bash scripts/retrain_gpu.sh
```
