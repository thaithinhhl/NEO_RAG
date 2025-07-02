from docx import Document 
import unicodedata
import re 
import os

'''convert docx to text'''
def extract_text_from_docx(file_path: str) -> str:
    doc  = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

'''chuan hoa unicode'''
def normalize_text(text: str) -> str:
    return unicodedata.normalize('NFC', text) 

'''loai bo ky tu thua '''
def clean_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\-+', '', text)
    # text = re.sub(r'\s+', ' ', text)  
    text  = text.strip()
    return text 

'''data_new.txt'''
def save_to_file(text: str, output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Save_file: {output_path}")

if __name__ == '__main__':
    file_path = 'data/Legan.docx'
    output_file = 'data/Legan_new.txt'
    text = extract_text_from_docx(file_path)
    text = normalize_text(text)
    text = clean_text(text) 
    save_to_file(text, output_file) 

