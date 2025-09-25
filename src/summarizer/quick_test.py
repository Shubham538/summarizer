from pathlib import Path
from tools.utils import extract_pdf, count_tokens, recursive_chunking


BASE_DIR = Path(__file__).resolve().parent.parent.parent  
file_path = BASE_DIR / "src" /"summarizer"/ "inputs" / "SRN-010382.pdf"

raw_text = extract_pdf(str(file_path), "extracted_text.txt")
chunks = recursive_chunking(raw_text)
print(chunks)

print( "token count =", count_tokens(chunks))


