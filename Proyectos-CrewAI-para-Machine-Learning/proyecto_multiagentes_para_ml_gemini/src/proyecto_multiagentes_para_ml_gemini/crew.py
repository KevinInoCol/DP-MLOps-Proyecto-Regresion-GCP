from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from proyecto_multiagentes_para_ml_gemini.tools.csv_profiler_tool import CSVProfilerTool
from proyecto_multiagentes_para_ml_gemini.tools.notebook_writer_tool import NotebookWriterTool
from proyecto_multiagentes_para_ml_gemini.tools.file_reader_tool import FileReaderTool


@CrewBase
class ProyectoAgentesDePipelineMl():
    """Crew de agentes especializados para construir pipelines de ML para cualquier dataset tabular."""

    agents: List[BaseAgent]
    tasks: List[Task]

    # --- Agentes ---

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['data_analyst'],  # type: ignore[index]
            tools=[CSVProfilerTool(), NotebookWriterTool()],
            verbose=True
        )

    @agent
    def feature_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['feature_engineer'],  # type: ignore[index]
            tools=[CSVProfilerTool(), NotebookWriterTool()],
            verbose=True
        )

    @agent
    def pipeline_builder(self) -> Agent:
        return Agent(
            config=self.agents_config['pipeline_builder'],  # type: ignore[index]
            tools=[NotebookWriterTool(), FileReaderTool()],
            verbose=True,
            max_iter=50,
        )

    @agent
    def feature_selector(self) -> Agent:
        return Agent(
            config=self.agents_config['feature_selector'],  # type: ignore[index]
            tools=[NotebookWriterTool()],
            verbose=True
        )

    @agent
    def model_selector(self) -> Agent:
        return Agent(
            config=self.agents_config['model_selector'],  # type: ignore[index]
            tools=[NotebookWriterTool()],
            verbose=True
        )

    @agent
    def model_trainer(self) -> Agent:
        return Agent(
            config=self.agents_config['model_trainer'],  # type: ignore[index]
            tools=[NotebookWriterTool()],
            verbose=True
        )

    @agent
    def scorer(self) -> Agent:
        return Agent(
            config=self.agents_config['scorer'],  # type: ignore[index]
            tools=[NotebookWriterTool()],
            verbose=True
        )

    @agent
    def final_pipeline_builder(self) -> Agent:
        return Agent(
            config=self.agents_config['final_pipeline_builder'],  # type: ignore[index]
            tools=[NotebookWriterTool()],
            verbose=True,
            max_iter=50,
        )

    @agent
    def code_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config['code_reviewer'],  # type: ignore[index]
            tools=[FileReaderTool(), NotebookWriterTool()],
            verbose=True
        )

    # --- Tareas ---

    @task
    def eda_task(self) -> Task:
        return Task(
            config=self.tasks_config['eda_task'],  # type: ignore[index]
        )

    @task
    def feature_engineering_task(self) -> Task:
        return Task(
            config=self.tasks_config['feature_engineering_task'],  # type: ignore[index]
        )

    @task
    def pipeline_task(self) -> Task:
        return Task(
            config=self.tasks_config['pipeline_task'],  # type: ignore[index]
            context=[self.feature_engineering_task()],
        )

    @task
    def feature_selection_task(self) -> Task:
        return Task(
            config=self.tasks_config['feature_selection_task'],  # type: ignore[index]
        )

    @task
    def model_selection_task(self) -> Task:
        return Task(
            config=self.tasks_config['model_selection_task'],  # type: ignore[index]
            context=[self.eda_task(), self.feature_selection_task()],
        )

    @task
    def model_training_task(self) -> Task:
        return Task(
            config=self.tasks_config['model_training_task'],  # type: ignore[index]
            context=[self.pipeline_task(), self.feature_selection_task()],
        )

    @task
    def scoring_task(self) -> Task:
        return Task(
            config=self.tasks_config['scoring_task'],  # type: ignore[index]
            context=[
                self.feature_engineering_task(),
                self.pipeline_task(),
                self.feature_selection_task(),
                self.model_training_task(),
            ],
        )

    @task
    def final_pipeline_task(self) -> Task:
        return Task(
            config=self.tasks_config['final_pipeline_task'],  # type: ignore[index]
            context=[
                self.pipeline_task(),
                self.feature_selection_task(),
                self.model_training_task(),
            ],
        )

    @task
    def code_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['code_review_task'],  # type: ignore[index]
            context=[
                self.eda_task(),
                self.feature_engineering_task(),
                self.pipeline_task(),
                self.feature_selection_task(),
                self.model_selection_task(),
                self.model_training_task(),
                self.scoring_task(),
                self.final_pipeline_task(),
            ]
        )

    # --- Crew ---

    @crew
    def crew(self) -> Crew:
        """Crew secuencial para generar un pipeline de ML adaptativo para cualquier dataset."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
