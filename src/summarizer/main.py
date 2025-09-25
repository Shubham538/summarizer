from pathlib import Path
import warnings
from summarizer.crew import Summarizer
from dotenv import load_dotenv
from tools.render_markdown import render_markdown_html
from tools.utils import extract_pdf, recursive_chunking
from tools.token_counter import estimate_crew_run_tokens
import yaml

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

load_dotenv(override=True)


def run(base_name: str):
    """
    Run the crew.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent.parent  
    file_path = BASE_DIR / "src" /"summarizer"/ "inputs" / f"{base_name}.pdf"
    
    raw_text = extract_pdf(str(file_path), "extracted_text.txt")

    chunks = recursive_chunking(raw_text) #print(chunks) 
          
    # load tasks and agents configs
    tasks_cfg = yaml.safe_load(Path(BASE_DIR/"src/summarizer/config/tasks.yaml").read_text())
    agents_cfg = yaml.safe_load(Path(BASE_DIR/"src/summarizer/config/agents.yaml").read_text())

    task_prompts = [v.get("description", "expected_output") for v in tasks_cfg.values() if isinstance(v, dict)]
    agent_prompts = []
    for v in agents_cfg.values():
        if isinstance(v, dict):
            agent_prompts.extend(
                [v.get(f) for f in ("role", "goal", "backstory", "system", "instructions") if v.get(f)]
            )
    
    inputs = {'chunks': chunks}

    # Gemini exact count (calls API)
    est_gemini = estimate_crew_run_tokens(
        inputs_chunks=chunks,
        task_prompts=task_prompts,
        agent_system_prompts=agent_prompts,
        model="gemini-2.0-flash",
        provider="gemini",
    )
    
    print("[Estimator:Gemini] Breakdown:", est_gemini)
    print("[Estimator:Gemini] Grand total tokens =", est_gemini["grand_total"])

    crew_instance = Summarizer()
    result = crew_instance.crew().kickoff(inputs=inputs)

    print("Research completed!!")

    md_file_path = BASE_DIR / "research_findings.txt"

    html = render_markdown_html(md_file_path)
    Path("result.html").write_text(html, encoding="utf-8")

    return result


# def openai_est():
#     """Use the following when using OpenAI"""

#     # OpenAI estimate (tiktoken, offline)
#     est_openai = estimate_crew_run_tokens(
#         inputs_chunks=chunks,
#         task_prompts=task_prompts,
#         agent_system_prompts=agent_prompts,
#         model="gpt-4o-mini",
#         provider="openai",
#     )

#     print("[Estimator:OpenAI] Breakdown:", est_openai)
#     print("[Estimator:OpenAI] Grand total tokens =", est_openai["grand_total"])


