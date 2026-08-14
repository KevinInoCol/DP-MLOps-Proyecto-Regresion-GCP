---
name: install-uv
description: Instala uv (el gestor de paquetes y entornos de Python de Astral) detectando automáticamente el sistema operativo — macOS, Linux, WSL o Windows. Úsala cuando el usuario pida instalar uv, cuando un comando falle con "uv - command not found", o cuando el usuario mencione "instalar uv", "no tengo uv", "/install-uv".
---

Instala `uv` en la máquina del usuario. Sigue estos pasos **en orden** y no los saltes.

## Paso 1 — ¿Ya está instalado?

```bash
uv --version
```

- **Si responde con una versión** (ej. `uv 0.10.9`): ya está instalado. Muestra la versión y la ruta (`which uv`, o `where uv` en Windows CMD), dile al usuario que no hace falta hacer nada, y ofrece actualizarlo con `uv self update`. **Termina aquí.**
- **Si dice `command not found` / `no se reconoce`**: continúa al Paso 2.

## Paso 2 — Detectar el sistema operativo

```bash
uname -s
```

Interpreta la salida:

| Salida | Sistema | Instalador |
|---|---|---|
| `Darwin` | macOS | script `sh` |
| `Linux` | Linux o WSL | script `sh` |
| `MINGW*`, `MSYS*`, `CYGWIN*` | Windows (Git Bash) | script PowerShell |
| el comando falla | Windows (PowerShell/CMD nativo) | script PowerShell |

## Paso 3 — Instalar

### macOS y Linux (incluye WSL)

En macOS, si el usuario ya usa Homebrew, `brew install uv` es válido y más fácil de actualizar. Pregúntale cuál prefiere solo si detectas `brew`; si no, usa el instalador oficial:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Si el usuario está en PowerShell nativo (no Git Bash), el comando es solo la parte interna:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

## Paso 4 — Verificar

El instalador coloca el binario en `~/.local/bin` (macOS/Linux) o `%USERPROFILE%\.local\bin` (Windows), y esa carpeta **todavía no está en el PATH de la terminal actual**. Verifica con la ruta completa:

```bash
~/.local/bin/uv --version
```

Si eso funciona, la instalación fue exitosa. Explícale al usuario que para poder escribir solo `uv` debe:

- **macOS / Linux:** ejecutar `source $HOME/.local/bin/env` en la terminal actual, **o** cerrar y volver a abrir la terminal.
- **Windows:** cerrar y volver a abrir la terminal (PowerShell o Git Bash).

Después de reiniciar la terminal, que confirme con:

```bash
uv --version
```

## Notas

- **No uses `pip install uv` ni `brew` en Linux.** El instalador oficial es un binario autocontenido que no depende de ningún Python previo — que es justamente el punto de uv.
- **No hace falta instalar Python antes.** uv descarga la versión de Python que cada proyecto necesite (`uv venv --python 3.12`).
- Si `curl` no existe en Linux, usa la alternativa con wget:
  ```bash
  wget -qO- https://astral.sh/uv/install.sh | sh
  ```
- Si el usuario pregunta dónde se instaló: uv es **un solo binario a nivel de usuario**, no por proyecto. Se instala una vez en la computadora y sirve para todos los proyectos.
