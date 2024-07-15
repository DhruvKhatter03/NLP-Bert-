import fitz  

pdf_path = 'data/textbooks/textbook1.pdf'

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            text += page.get_text()
    return text

text_content = extract_text_from_pdf(pdf_path)
print("Text content of textbook1.pdf:")
print(text_content[:500])  #
