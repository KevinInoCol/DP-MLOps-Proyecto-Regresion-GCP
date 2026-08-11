import pandas as pd
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class CSVProfilerInput(BaseModel):
    """Input schema para CSVProfilerTool."""
    file_path: str = Field(..., description="Ruta al archivo CSV a perfilar.")


class CSVProfilerTool(BaseTool):
    name: str = "CSV Profiler Tool"
    description: str = (
        "Carga un archivo CSV y retorna un perfil completo del dataset: "
        "shape, tipos de variables, porcentaje de valores faltantes por columna, "
        "estadísticas descriptivas de variables numéricas, cardinalidad de variables "
        "categóricas y las primeras 5 filas. Úsala antes de generar cualquier notebook "
        "para entender qué transformaciones necesita el dataset."
    )
    args_schema: Type[BaseModel] = CSVProfilerInput

    def _run(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            return f"Error al cargar el CSV: {e}"

        lines = []

        # Shape
        lines.append(f"## Shape\nFilas: {df.shape[0]} | Columnas: {df.shape[1]}\n")

        # Tipos de variables
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        lines.append(f"## Tipos de variables")
        lines.append(f"Numéricas ({len(num_cols)}): {num_cols}")
        lines.append(f"Categóricas ({len(cat_cols)}): {cat_cols}\n")

        # Valores faltantes
        missing = df.isnull().mean().mul(100).round(2)
        missing_cols = missing[missing > 0].sort_values(ascending=False)
        if len(missing_cols) > 0:
            lines.append("## Valores faltantes (% por columna)")
            for col, pct in missing_cols.items():
                lines.append(f"  {col}: {pct}%")
        else:
            lines.append("## Valores faltantes\nNo hay valores faltantes.")
        lines.append("")

        # Estadísticas numéricas
        if num_cols:
            desc = df[num_cols].describe().round(3).to_string()
            lines.append(f"## Estadísticas descriptivas (numéricas)\n{desc}\n")

        # Cardinalidad categóricas
        if cat_cols:
            lines.append("## Cardinalidad de variables categóricas")
            for col in cat_cols:
                n_unique = df[col].nunique()
                top_vals = df[col].value_counts().head(5).to_dict()
                lines.append(f"  {col}: {n_unique} valores únicos | Top 5: {top_vals}")
        lines.append("")

        # Primeras filas
        lines.append(f"## Primeras 5 filas\n{df.head(5).to_string()}\n")

        # Variables con sesgo alto (candidatas a transformación log/yeo-johnson)
        skewed = df[num_cols].skew().abs().sort_values(ascending=False)
        high_skew = skewed[skewed > 0.75]
        if len(high_skew) > 0:
            lines.append("## Variables numéricas con sesgo alto (|skew| > 0.75) — candidatas a transformación")
            for col, sk in high_skew.items():
                lines.append(f"  {col}: skew = {round(sk, 3)}")
        lines.append("")

        # Variables con muchos ceros (candidatas a binarización)
        zero_dominated = {}
        for col in num_cols:
            zero_pct = (df[col] == 0).mean()
            if zero_pct > 0.5:
                zero_dominated[col] = round(zero_pct * 100, 1)
        if zero_dominated:
            lines.append("## Variables con >50% ceros — candidatas a binarización")
            for col, pct in zero_dominated.items():
                lines.append(f"  {col}: {pct}% ceros")

        return "\n".join(lines)
