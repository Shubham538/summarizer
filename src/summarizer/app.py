import gradio as gr
from summarizer.main import run



with gr.Blocks() as demo:
    gr.Markdown("Enter base filename (without .pdf)")
    name = gr.Textbox(label="Base filename(Case Sensitive)", placeholder="e.g., SRN-010382")
    out = gr.Textbox(label="Crew Output")
    gr.Button("Run").click(fn=run, inputs=name, outputs=out)
demo.launch()
