import json
import os
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class FileReaderInput(BaseModel):
    """Input schema para FileReaderTool."""
    file_path: str = Field(..., description="Ruta al archivo a leer.")


class FileReaderTool(BaseTool):
    name: str = "File Reader Tool"
    description: str = (
        "Lee el contenido de un archivo. Si es un Jupyter Notebook (.ipynb), "
        "extrae y retorna el código fuente y markdown de cada celda en formato legible. "
        "Si es un archivo de texto (.py, .md, .csv, etc.), retorna su contenido."
    )
    args_schema: Type[BaseModel] = FileReaderInput

    def _run(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return f"Error: el archivo '{file_path}' no existe."

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return f"Error al leer el archivo: {e}"

        if file_path.endswith(".ipynb"):
            return self._parse_notebook(content, file_path)

        return content

    def _parse_notebook(self, content: str, file_path: str) -> str:
        try:
            nb = json.loads(content)
        except json.JSONDecodeError:
            return f"Error: '{file_path}' no es un JSON válido. El notebook está corrupto."

        cells = nb.get("cells", [])
        if not cells:
            return f"El notebook '{file_path}' no tiene celdas."

        lines = [f"## Notebook: {file_path} ({len(cells)} celdas)\n"]

        for i, cell in enumerate(cells):
            cell_type = cell.get("cell_type", "unknown")
            source = cell.get("source", [])
            if isinstance(source, list):
                source_text = "".join(source)
            else:
                source_text = source

            lines.append(f"### Celda {i+1} ({cell_type})")
            lines.append(source_text)
            lines.append("")

        return "\n".join(lines)
