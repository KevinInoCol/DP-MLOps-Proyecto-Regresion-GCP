import json
import re
import os
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class NotebookWriterInput(BaseModel):
    """Input schema para NotebookWriterTool."""
    output_path: str = Field(..., description="Ruta completa donde guardar el notebook (.ipynb).")
    content: str = Field(
        ...,
        description=(
            "Contenido del notebook. Puede ser:\n"
            "1. Bloques de código Python entre triple backticks (```python ... ```) "
            "intercalados con texto markdown.\n"
            "2. Texto plano con secciones markdown y código indentado.\n"
            "El tool parseará el contenido y creará las celdas automáticamente."
        )
    )


class NotebookWriterTool(BaseTool):
    name: str = "Notebook Writer Tool"
    description: str = (
        "Crea y guarda un archivo Jupyter Notebook (.ipynb) válido a partir del contenido "
        "que le pases. Acepta texto con bloques de código Python entre triple backticks "
        "(```python ... ```) intercalados con texto markdown. Úsala para guardar cada "
        "notebook generado en su ruta de destino."
    )
    args_schema: Type[BaseModel] = NotebookWriterInput

    def _run(self, output_path: str, content: str) -> str:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        cells = self._parse_cells(content)

        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.11.0"
                }
            },
            "cells": cells
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)

        return f"Notebook guardado exitosamente en: {output_path} ({len(cells)} celdas)"

    def _parse_cells(self, content: str) -> list:
        """Parsea el contenido en celdas de código y markdown."""
        cells = []
        # Separar bloques de código Python del resto (markdown)
        pattern = r"```(?:python)?\n(.*?)```"
        parts = re.split(pattern, content, flags=re.DOTALL)

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            if i % 2 == 1:
                # Bloque de código
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": part.splitlines(keepends=True)
                })
            else:
                # Texto markdown
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": part.splitlines(keepends=True)
                })

        # Si no había backticks, todo el contenido va como una sola celda de código
        if not cells:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": content.splitlines(keepends=True)
            })

        return cells
