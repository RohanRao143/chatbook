# write an api using fastapi that serves a pdf chatbot

# use case 1: upload a pdf, process it, and create a vector database using faiss

# post file processing, return a unique id for the file and store the vector 
# database in an azure or aws s3 instance with the unique id as the file name

# once session started with the unique id fetch the vector database from s3
#  and use it to answer questions

# use case 2: ask questions to the pdf chatbot using the unique id to fetch the vector database


from fastapi import FastAPI, File, UploadFile, HTTPException, Request
# from cloudflare import Cloudflare
import boto3
from botocore.client import Config

from app.chatpdf import embed_query, get_index, summarize_semantics, get_index_name

def execute_after_delay(delay, callback_func):
    time.sleep(delay)
    callback_func()


from app.utilities import process_file
 
# import tabula
import pandas as pd

import os



# # Create a client to connect to Cloudflare's R2 Storage
# s3_client = boto3.client(
#     's3',
#     endpoint_url=ConnectionUrl,
#     aws_access_key_id=Access_key,
#     aws_secret_access_key=Secret_access,
#     config=Config(signature_version='s3v4'),
#     region_name='us-east-1'
# )


# def create_bucket_if_not_exists(bucket_name: str):
#     try:
#         bucket = client.r2.buckets.get(
#             account_id=ACCOUNT_ID,
#             bucket_name=bucket_name,
#         )
#         print(f"Bucket {bucket_name} already exists.")
#     except Exception as e:
#         print(f"Creating bucket {bucket_name}.")
#         bucket = client.r2.buckets.create(
#             account_id=ACCOUNT_ID,
#             name=bucket_name,
#         )
#     return bucket

# def upload_file(filename, bucket="chatbook", object_name=None):
#     # Upload to R2 using S3 compatible API
#     # rs = s3_upload(Bucket=bucket,
#     #         S3Client=S3Connect,
#     #         TargetFilePath=f"test/my file.json",
#     #         UploadObject=UploadObject,
#     #         UploadMethod="Object"
#     #     )
#     # If S3 object_name was not specified, use file_name
#     if object_name is None:
#         object_name = os.path.basename(filename)
    
#     try:
#         response = s3_client.upload_file(filename, bucket, object_name)
#         print("UPLOADED : " + filename + " upload success")
#     except Exception as e:
#         print(e)
#         return False
#     return True



# def download_file(filename, bucket="chatbook"):

#     try:
#         if os.path.isfile(f"./data/{filename}"):
#             pass
#         else:
#             # os.mkdir("data/", mode=0o777, dir_fd=None)
#             if not os.path.exists("./data"):
#                 os.makedirs("./data")
#             s3_client.download_file(bucket, filename, "./data")
#     except Exception as e:
#         try:
#             s3_client.download_file(bucket, filename, "./data")
#         except Exception as e:
#             pass
#     return None

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": {"theme": "obsidian"}})


def save_to_csv(knowledge_source: UploadFile):    
    try:
        contents = knowledge_source.file.read()
        try:
            if os.path.exists(f"data"):
                pass
            else:
                os.makedirs("data")
        except Exception as e:
            os.makedirs("data")
        with open(f"data/{knowledge_source.filename}", 'wb') as f:
            f.write(contents)
        # print(os.getcwd())
        # print(knowledge_source.filename)
        # print(os.listdir(os.getcwd()))

    except Exception:

        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        knowledge_source.file.close()

    # pdf_path = knowledge_source.filename # Replace with the path to your PDF file
    # csv_path = knowledge_source.filename + ".csv" # Desired output CSV file name
    # Convert all tables from all pages of the PDF to a single CSV file
    # tabula.convert_into(pdf_path, csv_path, output_format="csv", pages="all")
    # print(f"PDF tables from '{pdf_path}' converted to '{csv_path}'")
    


app = FastAPI(swagger_ui_parameters={"syntaxHighlight": {"theme": "obsidian"}})


def file_to_sentences(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, on_bad_lines="skip")
    print(data.shape)


def get_sentences(filename: str):
    df = pd.read_csv(f"data/{filename}.csv")
    sentences = None
    sentences = df["knowledge"].tolist()

    return sentences

import time
import shutil

def remove_cached_files():
    
    while True:
        time.sleep(999)
        shutil.rmtree('/data', ignore_errors=True)
        os.makedirs("./data")

    # os.remove(f"{filename}.csv")
    # index = get_index_name(filename)
    # os.remove(index)




@app.get("/chat")
async def read_root(request: Request, question: str, rel_docs: int, filename: str):
    # file_to_sentences("output.csv")
    # return { "question" : question }
    # if question is None:
    #   question = request.query_params.get("question", "Which Project Uses AI ?")
    # download_file(filename + ".csv")
    # download_file(get_index_name(filename))
    if (question):
        # filename = "the-future-is-faster-than-you-think.pdf"
        # filename = "MRohanRaoResumeProfile2025-1.pdf.pdf"
        # sentences = process_pdf_to_semantic_chunks(filename)
        index = get_index(filename)
        sentences = get_sentences(filename)
        xq = embed_query(question)
        k = 6

        if rel_docs < 10:
            k = rel_docs
        else:
            k = 9

        D, I = index.search(xq, k)  # search
        # for i in I[0]:
        #     print(sentences[i])
        context = [sentences[i] for i in I[0]]
        print(context)
        # return {"generative_response": google_summarizer(question, " ".join(context)), "question": question}
        # # return {"generative_response": summarize_text_t5(" ".join(context)), "question": question}
        return {"generative_response": summarize_semantics(" ".join(context)), "question": question}
    else:
        raise HTTPException(status_code=400, detail="Ask a question.")

# @app.post("/index_portfolio")
# async def index_data():
#     filename = "MRohanRaoResumeProfile2025-1.pdf.pdf"
#     try:
#         process_file(filename)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
#     return {"status": "Indexing completed successfully."}


@app.post("/upload_to_index/")
async def create_upload_file(knowledge_source: UploadFile = File(...)):

    if not knowledge_source.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    print("Received file:", knowledge_source)
    save_to_csv(knowledge_source)
    file_path = f"data/{knowledge_source.filename}"
    file_size_bytes = os.path.getsize(file_path)
    print(f"File size: {file_size_bytes} bytes")
    if file_size_bytes < 5000000:
        try:
            _, doc = process_file(knowledge_source.filename)
            # upload_file(doc)
            # index = get_index_name(knowledge_source.filename)
            # upload_file(index)

            # os.remove(doc)
            # os.remove(index)
            os.remove(f"data/{knowledge_source.filename}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
        # Process the PDF file and create vector database
        # Store the vector database in S3 with a unique id
        unique_id = "some_generated_unique_id"  # Replace with actual unique id generation logic
        return {"filename": knowledge_source.filename, "unique_id": unique_id, "status": "Indexing completed successfully."}
    else:
        raise HTTPException(status_code=400, detail="Upload file of size less than 5 MB.")



# @app.get("/")
# async def root():
#     return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)







# def process_faiss_index(filename: str):
#     sentences = process_pdf_to_semantic_chunks(filename)
#     index = get_index(filename)
#     if index is None:
#         print("Processing PDF to semantic chunks:")
#         sentence_embeddings = embed_sentences(sentences)
#         index = add_sentences_to_index(sentence_embeddings, filename)

#         print(sentences[:3])
#         data = { "knowledge": sentences }
#         df = pd.DataFrame(data)
#         df.to_csv(filename + ".csv", index=False)

#     return index, sentences


# @app.get("/")
# async def format_files(request: Request, question: str):
#     filename = "MRohanRaoResumeProfile2025-1.pdf.pdf"
#     # sentences = process_pdf_to_semantic_chunks(filename)

#     # sentences = get_sentences(filename)
#     # print(sentences[:5])

#     print(question)
#     return {}


# task = asyncio.create_task(remove_cached_files(knowledge_source.filename))




# Create a partial function with pre-filled arguments
# partial_callback = partial(remove_cached_files, knowledge_source.filename)
# execute_after_delay(999, partial_callback)


# process = Process(target=partial_callback)
# process.start()
# process.join(timeout=1000) # Wait for process to finish or timeout

# if process.is_alive():
#     print(f"Function timed out after {1000} seconds!")
#     process.terminate() # Terminate the process
#     process.join() # Wait for the process to actually terminate
# else:
#     print("Function completed within the timeout.")