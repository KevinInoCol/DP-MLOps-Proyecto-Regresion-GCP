# Comandos del Proyecto

Entorno virtual: **DP-MLOps-Regresion-GCP**
Python: **3.11** (los notebooks se ejecutaron con 3.11.15)

---

## 1. Crear el entorno virtual con conda

```bash
conda create -n DP-MLOps-Regresion-GCP python=3.11 -y
```

## 2. Activar el entorno virtual

```bash
conda activate DP-MLOps-Regresion-GCP
```

## 3. Instalar las dependencias desde requirements.txt

```bash
cd proyecto_regresion_casas
pip install -r requirements.txt
```

---

## Comandos útiles adicionales

### Registrar el kernel para usarlo en los notebooks

```bash
python -m ipykernel install --user --name DP-MLOps-Regresion-GCP --display-name "DP-MLOps-Regresion-GCP"
```

### Desactivar el entorno

```bash
conda deactivate
```

### Verificar la versión de Python y los paquetes instalados

```bash
python --version
pip list
```

### Eliminar el entorno (si necesitas recrearlo desde cero)

```bash
conda deactivate
conda env remove -n DP-MLOps-Regresion-GCP -y
```

### Listar todos los entornos de conda

```bash
conda env list
```
