# 📦 Transferir Repositorio a PC con GPU

Guía rápida para mover el proyecto de Mac a PC con RTX 4070.

---

## 1️⃣ En tu Mac - Preparar y Transferir

### Comprimir el repositorio (sin modelos pesados):

```bash
cd /Users/enriquegomeztagle/Desktop/0241823/UP/multi-agent-flocking-foraging-rl

# Crear un tar.gz limpio
tar -czf ~/Desktop/flocking-rl-gpu.tar.gz \
  --exclude="models/ppo_*" \
  --exclude="results/*.json" \
  --exclude="*.pyc" \
  --exclude="__pycache__" \
  --exclude=".git" \
  .
```

**Resultado:** `~/Desktop/flocking-rl-gpu.tar.gz` (~2-5 MB sin modelos)

### Alternativa - Si quieres incluir modelos actuales como backup:

```bash
tar -czf ~/Desktop/flocking-rl-full.tar.gz \
  --exclude="*.pyc" \
  --exclude="__pycache__" \
  --exclude=".git" \
  .
```

**Resultado:** `~/Desktop/flocking-rl-full.tar.gz` (~70 MB con modelos)

---

## 2️⃣ Transferir a PC

Opciones (elige la más rápida):

### Opción A: USB
```bash
# Copiar a USB
cp ~/Desktop/flocking-rl-gpu.tar.gz /Volumes/USB_NAME/
```

### Opción B: Red local (si están en la misma red)
```bash
# En Mac (enviar)
scp ~/Desktop/flocking-rl-gpu.tar.gz usuario@PC_IP:~/Downloads/
```

### Opción C: AirDrop / Dropbox / Google Drive
Arrastra el archivo `flocking-rl-gpu.tar.gz` desde Desktop

---

## 3️⃣ En tu PC con GPU - Setup

### Descomprimir:

```bash
# Windows (Git Bash / WSL / PowerShell)
cd C:\Users\TU_USUARIO\Documents
mkdir flocking-rl
cd flocking-rl
tar -xzf "C:\Users\TU_USUARIO\Downloads\flocking-rl-gpu.tar.gz"

# Linux
cd ~/projects
mkdir flocking-rl
cd flocking-rl
tar -xzf ~/Downloads/flocking-rl-gpu.tar.gz
```

---

## 4️⃣ Instalar Dependencias (si es primera vez en este PC)

### Verificar Python y GPU:

```bash
python --version  # Debe ser 3.8+
nvidia-smi        # Debe mostrar RTX 4070
```

### Crear entorno virtual e instalar:

```bash
# Crear venv
python -m venv .venv

# Activar
# Windows:
.venv\Scripts\activate
# Linux:
source .venv/bin/activate

# Instalar PyTorch con CUDA (importante!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar resto de dependencias
pip install -r requirements.txt
```

### Verificar CUDA en Python:

```bash
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Esperado:**
```
CUDA disponible: True
GPU: NVIDIA GeForce RTX 4070
```

---

## 5️⃣ Ejecutar Reentrenamiento

### Verificación rápida:

```bash
# Windows (Git Bash / WSL):
bash scripts/retrain_gpu.sh

# Windows (PowerShell):
python train/train_ppo.py --config configs/env_hard.yaml --output models/ppo_hard --timesteps 4000000
```

### El script hará:

1. ✅ Verificar GPU
2. 🔄 Pedir confirmación
3. 🚀 Entrenar Hard Mode (4M steps, ~40-60 min)
4. 📊 Evaluar Hard Mode (100 eps, ~3-5 min)
5. 🚀 Entrenar Medium Mode (3M steps, ~30-40 min)
6. 📊 Evaluar Medium Mode (100 eps, ~3-5 min)

**Tiempo total: ~2 horas**

---

## 6️⃣ Durante el Entrenamiento

### Monitorear GPU:

```bash
# En otra terminal
nvidia-smi -l 1  # Actualiza cada segundo
```

**Esperado:**
- GPU Utilization: 70-95%
- Memory Used: ~4-6 GB
- Temperature: 60-75°C

### Si algo falla:

Ver logs en tiempo real:
```bash
tail -f models/ppo_hard/training.log  # Si existe
```

---

## 7️⃣ Después del Entrenamiento

### Comparar resultados:

```bash
python scripts/compare_retraining.py
```

### Transferir resultados de vuelta a Mac:

```bash
# En PC: Comprimir solo resultados nuevos
tar -czf results_gpu.tar.gz \
  models/ppo_hard/ \
  models/ppo_medium/ \
  results/hard_evaluation_4M.json \
  results/medium_evaluation_3M.json

# Transferir a Mac (SCP, USB, etc.)
```

---

## 🚨 Troubleshooting

### "CUDA out of memory"
```bash
# Editar train_ppo.py, reducir n_envs:
# Línea ~334: default=4 → default=2
```

### Script no ejecuta en Windows
```bash
# Usar comando directo en PowerShell:
python train/train_ppo.py --config configs/env_hard.yaml --output models/ppo_hard --timesteps 4000000
```

### PyTorch no detecta GPU
```bash
# Reinstalar PyTorch con CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## ⏱️ Timeline Esperado

```
00:00 - Setup y verificación (5 min)
00:05 - Hard Mode training inicia
00:50 - Hard Mode completo, evaluación
00:55 - Medium Mode training inicia
01:30 - Medium Mode completo, evaluación
01:35 - Comparación de resultados
01:40 - LISTO ✅
```

---

## 📋 Checklist Rápido

- [ ] Archivo transferido a PC
- [ ] Descomprimido correctamente
- [ ] Python 3.8+ instalado
- [ ] PyTorch con CUDA instalado
- [ ] `nvidia-smi` muestra RTX 4070
- [ ] `torch.cuda.is_available()` devuelve True
- [ ] Ejecutar: `bash scripts/retrain_gpu.sh`
- [ ] Monitorear GPU con `nvidia-smi -l 1`
- [ ] Esperar ~2 horas
- [ ] Comparar resultados
- [ ] Transferir de vuelta a Mac

---

**¡Listo para GPU training! 🚀**
