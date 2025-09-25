from pathlib import Path
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_pdf(pdf_path, output_file="extracted_text.txt"):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text())
    final_text = "\n".join(filter(None, text))
    Path(output_file).write_text(final_text, encoding="utf-8")
    return final_text
 
def recursive_chunking(
    text: str,
    tokens_per_chunk: int = 800,
    token_overlap: int = 200,
    model_name: str = "gpt-4o-mini"
) -> list[str]:
    """Recursive character chunking."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=model_name,
        chunk_size=tokens_per_chunk,
        chunk_overlap=token_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        keep_separator=False,
        strip_whitespace=True,
        add_start_index=True,
    )
    return splitter.split_text(text)

