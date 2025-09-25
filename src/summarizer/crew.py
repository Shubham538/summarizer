from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class Summarizer():
    """Summarizer crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True
        )


    @task
    def research_updates(self) -> Task:
        return Task(
            config=self.tasks_config['research_updates'],
            output_file='research_findings.txt'
        )


    @crew
    def crew(self) -> Crew:
        """Creates the Summarizer crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    def kickoff(self, inputs):
        """
        Run the crew with the provided inputs.
        This method ensures inputs are properly passed to the crew.
        """
        return self.crew().kickoff(inputs=inputs)