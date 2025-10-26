# write an api using fastapi that serves a pdf chatbot

# use case 1: upload a pdf, process it, and create a vector database using faiss

# post file processing, return a unique id for the file and store the vector 
# database in an azure or aws s3 instance with the unique id as the file name

# once session started with the unique id fetch the vector database from s3
#  and use it to answer questions

# use case 2: ask questions to the pdf chatbot using the unique id to fetch the vector database


from fastapi import FastAPI, File, UploadFile, HTTPException, Request
# from cloudflare import Cloudflare

from chatpdf import process_pdf_to_semantic_chunks, embed_sentences, add_sentences_to_index, embed_query, get_index, generate_from_semantics, summarize_semantics

import tabula
import pandas as pd

# client = Cloudflare(
#     api_token="WhCl1yjbgG-upSsEHjD8d0LAQ6fPICiNEXkGHdgX",
# )

# def create_bucket_if_not_exists(bucket_name: str):
#     try:
#         bucket = client.r2.buckets.get(
#             account_id="817a0f396e90e04c6d018878d7b1ce57",
#             bucket_name=bucket_name,
#         )
#         print(f"Bucket {bucket_name} already exists.")
#     except Exception as e:
#         print(f"Creating bucket {bucket_name}.")
#         bucket = client.r2.buckets.create(
#             account_id="817a0f396e90e04c6d018878d7b1ce57",
#             name=bucket_name,
#         )
#     return bucket   

def save_to_csv(knowledge_source: UploadFile):    
    try:
        contents = knowledge_source.file.read()
        with open(knowledge_source.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        knowledge_source.file.close()
    pdf_path = knowledge_source.filename # Replace with the path to your PDF file
    csv_path = knowledge_source.filename + ".csv" # Desired output CSV file name
    # Convert all tables from all pages of the PDF to a single CSV file
    tabula.convert_into(pdf_path, csv_path, output_format="csv", pages="all")
    print(f"PDF tables from '{pdf_path}' converted to '{csv_path}'")
    


app = FastAPI(swagger_ui_parameters={"syntaxHighlight": {"theme": "obsidian"}})


def file_to_sentences(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, on_bad_lines="skip")
    print(data.shape)


def get_sentences(filename: str):
    df = pd.read_csv(filename+".csv")
    sentences = None
    sentences = df["knowledge"].tolist()

    return sentences



def process_faiss_index(filename: str):
    sentences = process_pdf_to_semantic_chunks(filename)
    index = get_index(filename)
    if index is None:
        print("Processing PDF to semantic chunks:")
        sentence_embeddings = embed_sentences(sentences)
        index = add_sentences_to_index(sentence_embeddings, filename)

        print(sentences[:3])
        data = { "knowledge": sentences }
        df = pd.DataFrame(data)
        df.to_csv(filename + ".csv", index=False)

    return index, sentences


@app.get("/")
async def format_files(request: Request, question: str):
    filename = "MRohanRaoResumeProfile2025-1.pdf.pdf"
    # sentences = process_pdf_to_semantic_chunks(filename)

    # sentences = get_sentences(filename)
    # print(sentences[:5])

    print(question)
    return {}

            

@app.get("/chat")
async def read_root(request: Request, question: str):
    # file_to_sentences("output.csv")
    # print(question)
    # return { "question" : question }
    if question is None:
      question = request.query_params.get("question", "Which Project Uses AI ?")
    if (question):
        filename = "the-future-is-faster-than-you-think.pdf"
        filename = "MRohanRaoResumeProfile2025-1.pdf.pdf"
        # sentences = process_pdf_to_semantic_chunks(filename)
        index = get_index(filename)
        sentences = get_sentences(filename)
        xq = embed_query(question)
        k = 6
        D, I = index.search(xq, k)  # search
        print(I, len(sentences))
        # for i in I[0]:
        #     print(sentences[i])
        context = [sentences[i] for i in I[0]]
        print(context)
        return {"response": summarize_semantics(" ".join(context)), "question": question}
    else:
        raise HTTPException(status_code=400, detail="Ask a question.")

@app.post("/index_portfolio")
async def index_data():
    filename = "MRohanRaoResumeProfile2025-1.pdf.pdf"
    try:
        process_faiss_index(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
    return {"status": "Indexing completed successfully."}


@app.post("/uploadfile/")
async def create_upload_file(knowledge_source: UploadFile = File(...)):

    print("Received file:", knowledge_source)
    if not knowledge_source.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    save_to_csv(knowledge_source)
    
    # Process the PDF file and create vector database
    # Store the vector database in S3 with a unique id
    unique_id = "some_generated_unique_id"  # Replace with actual unique id generation logic

    return {"filename": knowledge_source.filename, "unique_id": unique_id}

