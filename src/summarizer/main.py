#!/usr/bin/env python
import sys
from pathlib import Path
import warnings
from summarizer.crew import Summarizer
from dotenv import load_dotenv
import pdfplumber
import gradio as gr
import tiktoken
from tools.render_markdown import render_markdown_html

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

load_dotenv(override=True)

def extract_pdf(pdf_path, output_file="extracted_text.txt"):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text())
    final_text = "\n".join(filter(None, text))
    Path(output_file).write_text(final_text, encoding="utf-8")
    return final_text

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Estimate number of tokens in the given text for a specific model.
    Default: gpt-4o-mini.
    """
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def run(base_name: str):
    """
    Run the crew.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent.parent  
    file_path = BASE_DIR / "src" /"summarizer"/ "inputs" / f"{base_name}.pdf"
    raw_text = extract_pdf(str(file_path), "extracted_text.txt")
    
    print("Estimated Token Count:", count_tokens(raw_text))

    inputs = {
        'raw_text' : raw_text
    }
    
    crew_instance = Summarizer()
    result = crew_instance.crew().kickoff(inputs=inputs)

    print("Research completed!!")

    md_file_path = BASE_DIR / "research_findings.txt"

    html = render_markdown_html(md_file_path)
    Path("result.html").write_text(html, encoding="utf-8")

    return result


