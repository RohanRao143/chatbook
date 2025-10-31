import os

from fastapi import UploadFile, HTTPException
from app.chatpdf import embed_sentences, add_sentences_to_index

import pandas as pd

# Process PDF file into sentences and store csv and index in cloudflare



import pymupdf
import re


def save_to_csv(knowledge_source: UploadFile):    
    try:
        contents = knowledge_source.file.read()
        with open(knowledge_source.filename, 'wb') as f:
            f.write(contents)

        print(os.getcwd())
        print(knowledge_source.filename)
        print(os.listdir(os.getcwd()))
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        knowledge_source.file.close()

    print(os.listdir())
    # pdf_path = knowledge_source.filename # Replace with the path to your PDF file
    # csv_path = knowledge_source.filename + ".csv" # Desired output CSV file name
    # Convert all tables from all pages of the PDF to a single CSV file
    # tabula.convert_into(pdf_path, csv_path, output_format="csv", pages="all")
    # print(f"PDF tables from '{pdf_path}' converted to '{csv_path}'")




def process_faiss_index(sentences, filename):
    # sentences = process_pdf_to_semantic_chunks(filename)
    # print("Processing PDF to semantic chunks:")
    
    sentence_embeddings = embed_sentences(sentences)
    index = add_sentences_to_index(sentence_embeddings, filename)
    return index

def file_to_chunks(file):
    doc = pymupdf.open(file) # open a document
    sentences = []
    for page in doc: # iterate the document pages
        text = page.get_text().encode("utf8") # get plain text (is in UTF-8)
        text = str(text).split("\\n")
        # re.sub(r'[^a-zA-Z0-9]', ' ', _.strip()) 
        data = [_.strip() for _ in text if len(_) > 4]
        if len(data) > 0:
            sentences += data
    return sentences


def process_file(filename):
    sentences = file_to_chunks(filename)
    if len(sentences) != 0:
        # index = get_index(filename)
        # if index is None:
        index = process_faiss_index(sentences, filename)
        # print(sentences[:3])
        data = { "knowledge": sentences }
        df = pd.DataFrame(data)
        df.to_csv(filename + ".csv", index=False)
        return index

    return None


