# To build a project that showcases a search-based system using a vector database, you don't need to fine-tune an LLM unless you're aiming for advanced tasks like question answering or summarization. Here's a **simplified approach**:

# Retrieval

# 1. **Store documents** in a vector database (e.g., FAISS, Chroma, or Pinecone) using embeddings from a pre-trained model (e.g., BERT, Sentence-BERT, or OpenAI's embeddings).
# 2. **Search** by converting your query into an embedding and finding the most similar vectors in the database.
# 3. **Retrieve** and display the most relevant document(s) from the search results.

# Augument and Generation.

# 4. build RAG pipeline to refine the results into human readable format.
#  Language Model for summarization. - Current or pre-trained QA model 


# For a **decent project**, consider the following **requirements**:

# - **Backend**: Python with Flask or FastAPI for handling requests.
# - **Vector database**: FAISS or Chroma for efficient similarity search.
# - **Embedding model**: Use Hugging Face's `sentence-transformers` or a pre-trained model.
# - **Frontend (optional)**: A simple web interface using HTML/CSS/JavaScript or React.
# - **Document processing**: Use libraries like `PyPDF2`, `docx`, or `pdfplumber` to extract text from documents.
# - **Search interface**: A text box for users to input queries and display results.
# - **Environment management**: Use `virtualenv` or `conda` to manage dependencies.







# Details

# IndexFlatL2 is a specific type of index in the FAISS library designed for efficient similarity search using L2 (Euclidean) distance.

#  When you use IndexFlatL2, the process will calculate l2 or euclidean distance between the question embeddings and the
#  document embeddings in the vector database to find the most relevant documents.



import pandas as pd


from sentence_transformers import SentenceTransformer
import faiss

# from transformers import T5Tokenizer, T5ForConditionalGeneration

from transformers import pipeline

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



model = SentenceTransformer('all-MiniLM-L6-v2') # use any other pre-trained model as needed

def file_to_sentences(file_path):

    data = pd.read_csv(file_path, sep='\t', header=None, on_bad_lines="skip")


def embed_sentences(sentences):
    sentence_embeddings = model.encode(sentences)
    # try:
    #     print(sentence_embeddings[0])
    # except Exception as e:
    #     print(f"Error printing embedding: {e}")

    # store these embeddings in vector database and separate is as different function
    return sentence_embeddings


def embed_query(query):
    query_embedding = model.encode([query])
    return query_embedding


def add_sentences_to_index(sentence_embeddings, filename):

    print(f"Embeddings shape: {sentence_embeddings.shape}")

    dimension_b = sentence_embeddings.shape[1]  # dimension

    index = faiss.IndexFlatL2(dimension_b)  # build the index

    print(index.is_trained)  # check if the index is trained

    index.add(sentence_embeddings)  # add vectors to the index

    faiss.write_index(index, f"{filename}_index.faiss")
    print(f"FAISS index saved to {filename}")

    return index

def get_index(filename):
    index = None
    try:
        index = faiss.read_index(f"{filename}_index.faiss")
    except Exception as e:
        print(f"Error loading index: {e}")
        index = None
    return index


# Initialize a summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")





def summarize_semantics(combined_text):

    # Define the input prompt and task
    # input_text = "This is a sample lecture content. Please summarize it."
    task = "summarize"

    # Preprocess the input text
    # input_ids = tokenizer.encode(combined_text, return_tensors='pt')
    # task_ids = tokenizer.encode(task, return_tensors='pt')

    # Generate the summary
    # output = model.generate(input_ids, task_ids)

    # Print the generated summary
    # print(tokenizer.decode(output[0], skip_special_tokens=True))
    # return tokenizer.decode(output[0], skip_special_tokens=True)
    summary = ""
    # max_length=256
    summary = summarizer(combined_text, max_length=49, min_length=50, do_sample=False)
    return summary






"""

USE IT WHEN DINE TUNING DATA



# T5 modelini ve tokenizer'ı yükle
model_name = "imxly/t5-pegasus-small"
model_name = "google/mt5-small"

# model_name = "Lucas-Hyun-Lee/T5_small_lecture_summarization"





# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# text_model = T5ForConditionalGeneration.from_pretrained(model_name)
# tokenizer = T5Tokenizer.from_pretrained(model_name)



def google_summarizer(question, input):
    
    # 3. Prepare the input text
    # For summarization, a common prefix is "summarize: "
    input_text = "Topic: " + question + " summarize: " + input

    # 4. Tokenize the input text
    # return_tensors="pt" ensures PyTorch tensors are returned
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # 5. Generate the output (summary)
    # max_length controls the maximum length of the generated summary
    # num_beams controls the number of beams for beam search, often improving quality
    output_ids = summarizer_model.generate(input_ids, max_length=50)

    # 6. Decode the generated output back to text
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 7. Print the generated summary
    print(f"Original Text: {input_text.replace('summarize: ', '')}")
    print(f"Generated Summary: {summary}")

    return summary


"""

# def summarize_semantics(combined_text):
#     # input_text = "This is a sample lecture content. Please summarize it."
#     input_text = "summarize: " + combined_text
#     # Preprocess the input text
#     input_ids = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).input_ids
#     # task_ids = tokenizer.encode(task, return_tensors='pt')
#     # Generate the summary
#     # output = summarizer_model.generate(input_ids, task_ids)
#     # return tokenizer.decode(output[0], skip_special_tokens=True)

#     # Generate the summary
#     summary_ids = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
#     # Decode the generated summary
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)


#     return summary




# model_name = "t5-small"  # Alternatif: t5-base, t5-large

# tokenizer = T5Tokenizer.from_pretrained(model_name)
# summarizer_model = T5ForConditionalGeneration.from_pretrained(model_name)

# def summarize_text_t5(text):
#     # Özetleme için T5 modeline uygun formatta giriş oluşturma
#     input_text = "summarize: " + text
#     inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
#     # Modeli çalıştırarak özetleme yapma
#     summary_ids = text_model.generate(inputs["input_ids"], max_length=150, min_length=50, num_beams=4, early_stopping=True)
#     # Özetlenen metni çözümleme
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     print("T5 small model: ", summary)

#     return summary
























# UNUSED imports



# from unstructured.partition.auto import partition
# from llama_cpp import Llama
# import os
# from csv import writer







# UNUSED or alternate methods



# Initialize a summarization pipeline
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")



# Initialize the tokenizer and model
# imxly/t5-pegasus-small
# Lucas-Hyun-Lee/T5_small_lecture_summarization

# tokenizer = T5Tokenizer.from_pretrained('Lucas-Hyun-Lee/T5_small_lecture_summarization')
# model = T5ForConditionalGeneration.from_pretrained('Lucas-Hyun-Lee/T5_small_lecture_summarization')










# UNUSED OR OLD METHODS




# llm = Llama(
#     #   model_path="../models/7B/llama-model.gguf",
#       model_path="../models/tllama.gguf",
#       # n_gpu_layers=-1, # Uncomment to use GPU acceleration
#       # seed=1337, # Uncomment to set a specific seed
#       # n_ctx=2048, # Uncomment to increase the context window
# )


# def process_pdf_to_semantic_chunks(file):
#     print(os.getcwd() + "/" + file)
#     print(file)
#     print(os.path.exists(os.getcwd() + "/" + file))
#     blocks = partition(filename=os.getcwd() + "/" + file)
#     print(os.getcwd() + "/" + file)
    
#     sentences = []
#     for block in blocks:
#         # print(f"{block.category}: {block.text}")
#         sentences.append(block.text)
#     sentences = [sentence for sentence in list(set(sentences)) if type(sentence) is str and sentence != ""]
#     return sentences






# def summarize_semantics(combined_text):

#     # Define the input prompt and task
#     # input_text = "This is a sample lecture content. Please summarize it."
#     task = "summarize"

#     # Preprocess the input text
#     # input_ids = tokenizer.encode(combined_text, return_tensors='pt')
#     # task_ids = tokenizer.encode(task, return_tensors='pt')

#     # Generate the summary
#     # output = model.generate(input_ids, task_ids)

#     # Print the generated summary
#     # print(tokenizer.decode(output[0], skip_special_tokens=True))
#     # return tokenizer.decode(output[0], skip_special_tokens=True)
#     summary = ""
#     # max_length=256
#     # summary = summarizer(combined_text, max_length=49, min_length=50, do_sample=False)
#     return summary




# def generate_from_semantics(sentences, question):

#     prompt = f"Be a smart agent. A question would be asked to you and relevant information would be provided.\
#     Your task is to answer the question and use the information provided. Question - {question}.\
#     Relevant Information - {[sentence for sentence in sentences]}"

#     prompt = f"Question: {question} Answer: {'. '.join(sentences)}"

#     # output = llm(
#     #     prompt, # Prompt
#     #     max_tokens=256, # Generate up to 32 tokens, set to None to generate up to the end of the context window
#     #     stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
#     #     echo=True # Echo the prompt back in the output
#     # ) # Generate a completion, can also call create_completion

#     print(sentences)

#     # output = llm.create_chat_completion(
#     #     messages = [
#     #         {"role": "system", "content": "A question would be asked to you and relevant\
#     #           information would be provided."},
#     #         {
#     #             "role": "user",
#     #             "content": prompt
#     #         }
#     #     ],
#     #     # response_format={
#     #     #     "type": "json_object",
#     #     # },
#     #     # temperature=0.7,
#     #     max_tokens=128
#     # )
#     output = None
    
#     return output







# sentences = get_sentences(urls)

# print(sentences[411], sentences[4798], sentences[4694], sentences[4623])

# sentence_embeddings = embed_sentences(sentences)
# index = add_sentences_to_index(sentence_embeddings, "knowledge")
# xq = embed_query("interception children of wooden stands")

# k = 4
# D, I = index.search(xq, k)  # search

# print(I)

# # I = [[1057, 656, 2956, 837]]

# for i in I[0]:
#     print(sentences[i])






# urls = [
#     'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2012/MSRpar.train.tsv',
#     'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2012/MSRpar.test.tsv',
#     'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2012/OnWN.test.tsv',
#     'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2013/OnWN.test.tsv',
#     'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2014/OnWN.test.tsv',
#     'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2014/images.test.tsv',
#     'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2015/images.test.tsv'
# ]

# def get_sentences(urls):
#     res = requests.get('https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_train.txt')
#     # create dataframe
#     data = pd.read_csv(StringIO(res.text), sep='\t')
    
#     sentences1 = data['sentence_A'].tolist()
#     sentences2 = data['sentence_B'].tolist()

#     sentences = sentences1 + sentences2

#     for url in urls:
#         res = requests.get(url)
#         # extract to dataframe

#         data = pd.read_csv(StringIO(res.text), sep='\t', header=None, on_bad_lines="skip")
#         # add to columns 1 and 2 to sentences list
#         sentences.extend(data[1].tolist())
#         sentences.extend(data[2].tolist())

#     sentences = [sentence for sentence in list(set(sentences)) if type(sentence) is str]
#     print(f"Total unique sentences: {len(sentences)}")

#     return sentences

