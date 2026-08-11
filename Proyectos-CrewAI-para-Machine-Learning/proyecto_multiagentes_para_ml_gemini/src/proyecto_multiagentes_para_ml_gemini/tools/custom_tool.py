from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

#El contrato de entrada
class MyCustomToolInput(BaseModel):
    #Define qué parámetros acepta el tool y de qué tipo son.
    #El Field(...) con description es crítico porque el LLM lee esta descripción para saber qué valor pasarle.
    #Si la descripción es mala, el agente pasa valores incorrectos.
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

#El tool en sí
class MyCustomTool(BaseTool):
    #El identificador. El agente decide si usar el tool basándose en este nombre.
    name: str = "Name of my tool"
    #Le explica al agente para qué sirve el tool. El LLM lee esto para decidir cuándo usarlo. Si es vaga, el agente no lo usará cuando debería.
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    #Conecta el tool con su schema de entrada. CrewAI lo usa para validar que el agente pase los argumentos correctos antes de ejecutar _run.
    args_schema: Type[BaseModel] = MyCustomToolInput

    #La lógica real. Lo único que hace Python puro. .
    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."
