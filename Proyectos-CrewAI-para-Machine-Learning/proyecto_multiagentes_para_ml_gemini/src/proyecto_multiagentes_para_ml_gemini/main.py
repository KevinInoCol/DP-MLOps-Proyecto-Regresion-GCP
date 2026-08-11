#!/usr/bin/env python
import sys
import warnings
import os

from proyecto_multiagentes_para_ml_gemini.crew import ProyectoAgentesDePipelineMl

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Ejecuta el crew de agentes para generar el pipeline de ML.
    """
    # Crear carpeta de salida si no existe
    output_folder = "proyecto_pipeline_machine_learning_muestra"
    os.makedirs(output_folder, exist_ok=True)

    inputs = {
        "dataset_path": "data/train.csv",
        "target_variable": "SalePrice",
        "output_folder": output_folder,
    }

    try:
        ProyectoAgentesDePipelineMl().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Entrena el crew por un número determinado de iteraciones.
    """
    output_folder = "proyecto_pipeline_machine_learning_muestra"
    os.makedirs(output_folder, exist_ok=True)

    inputs = {
        "dataset_path": "data/train.csv",
        "target_variable": "SalePrice",
        "output_folder": output_folder,
    }
    try:
        ProyectoAgentesDePipelineMl().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Repite la ejecución del crew desde una tarea específica.
    """
    try:
        ProyectoAgentesDePipelineMl().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Testea la ejecución del crew y retorna los resultados.
    """
    output_folder = "proyecto_pipeline_machine_learning_muestra"
    os.makedirs(output_folder, exist_ok=True)

    inputs = {
        "dataset_path": "data/train.csv",
        "target_variable": "SalePrice",
        "output_folder": output_folder,
    }
    try:
        ProyectoAgentesDePipelineMl().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    run()