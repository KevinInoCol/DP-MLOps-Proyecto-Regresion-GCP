# Comandos — Proyecto CrewAI

Preparación del entorno para `proyecto_multiagentes_para_ml_gemini`.

## 0. Instalar uv (solo una vez por computadora)

Abre este repositorio en **Claude Code** y ejecuta:

```
/install-uv
```

Claude detecta tu sistema operativo (macOS, Linux, WSL o Windows), instala `uv` y te confirma que funcionó. No tienes que copiar comandos ni saber qué instalador te toca.

`uv` es **un solo binario a nivel de usuario**: se instala una vez en tu máquina y sirve para todos tus proyectos, como `git`. Tampoco necesitas instalar Python antes — uv descarga la versión que cada proyecto requiera.

Al terminar, **cierra y vuelve a abrir la terminal** y comprueba:

```bash
uv --version
```

---

> Para los pasos 1 al 3, ubícate dentro de la carpeta del proyecto:
>
> ```bash
> cd proyecto_multiagentes_para_ml_gemini
> ```

## 1. Crear el entorno virtual

```bash
uv venv --python 3.12
```

Crea la carpeta `.venv/` dentro del proyecto. Se usa Python 3.12 porque el `pyproject.toml` exige `>=3.10,<3.14`. Si no tienes esa versión, `uv` la descarga automáticamente.

## 2. Activar el entorno virtual

```bash
source .venv/bin/activate
```

Sabrás que funcionó porque el prompt de la terminal muestra `(proyecto_multiagentes_para_ml_gemini)` al inicio.

Para salir del entorno:

```bash
deactivate
```

## 3. Instalar las dependencias

```bash
uv sync
```

Instala `crewai[tools,google-genai]==1.14.1`, `pandas` y todas sus dependencias, **leyendo las versiones exactas del archivo `uv.lock`**. Por eso todos en la clase terminan con un entorno idéntico. Además instala el proyecto en modo editable, de forma que los cambios en `src/` se reflejan sin reinstalar nada.

> No uses `uv pip install -e .` aquí: ese comando ignora el `uv.lock` y resuelve las versiones de cero, así que puedes terminar con dependencias distintas a las del resto de la clase.

---

## Todo junto (pasos 1 al 3)

Con uv ya instalado:

```bash
cd proyecto_multiagentes_para_ml_gemini
uv venv --python 3.12
source .venv/bin/activate
uv sync
crewai run
```

---

## Alternativa sin uv (con pip)

Si por alguna razón no puedes instalar uv, en esta misma carpeta tienes un `requirements.txt` con las versiones exactas exportadas del `uv.lock`. Necesitas tener **Python 3.10 a 3.13** ya instalado en tu sistema.

```bash
cd proyecto_multiagentes_para_ml_gemini
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r ../requirements.txt
pip install -e . --no-deps
crewai run
```

Dos detalles de esta ruta:

- El `requirements.txt` está **un nivel arriba**, junto a este archivo, de ahí el `../`.
- La segunda línea de `pip install` registra el propio proyecto como paquete (lo que `uv sync` hace solo). El flag `--no-deps` evita que pip vuelva a resolver dependencias y rompa los pines del paso anterior.

Es más lento y más frágil que `uv sync`. Úsalo solo como plan B.