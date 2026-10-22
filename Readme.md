# Chatbook — PDF Retrieval-Augmented Generation

Chatbook is a Python-based Retrieval-Augmented Generation (RAG) application for asking questions about uploaded PDF documents and generating content grounded in relevant passages from those documents.

Instead of relying only on a language model's pre-trained knowledge, a RAG pipeline retrieves source text at query time and supplies it as context for response generation.

## Key features

- Accept PDF documents as input.
- Extract document text for processing.
- Split extracted text into smaller chunks suitable for retrieval.
- Retrieve passages relevant to a user's prompt.
- Construct a context-aware prompt for a language model.
- Generate an answer or document-oriented content based on the retrieved context.

The exact UI, model provider, storage engine, and citation support depend on the repository configuration.

## RAG pipeline

```text
PDF upload
    ↓
Text extraction
    ↓
Text cleaning and chunking
    ↓
Indexing / retrieval preparation
    ↓
User question
    ↓
Relevant passage retrieval
    ↓
Context construction
    ↓
LLM response generation
    ↓
Answer to the user
```

## How it works

1. **Ingestion:** Load the selected PDF and extract its text.
2. **Preprocessing:** Normalize text and divide it into chunks while preserving useful context.
3. **Indexing:** Prepare chunks for the configured retrieval method, such as keyword search or vector embeddings.
4. **Query processing:** Accept a natural-language question or content-generation prompt.
5. **Retrieval:** Select the passages most relevant to the question.
6. **Context construction:** Add the retrieved passages and response instructions to the model prompt.
7. **Generation:** Ask the configured language model to produce a response grounded in the supplied context.
8. **Presentation:** Return the generated content to the user.

## Technology

- Python
- PDF text extraction
- Retrieval-Augmented Generation
- Large Language Models
- Text preprocessing and chunking
- Information retrieval
- Prompt construction

Add the exact libraries and services used by the codebase—for example, a specific PDF parser, embedding model, vector store, or LLM provider—rather than listing dependencies that are not installed.

## Setup

The repository's dependency file and entry point were not provided. For a Python project, a common setup pattern is:

```bash
python -m venv .venv
```

Activate the environment, then install dependencies using the file supplied by the project. For example, if the repository contains `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Start the application using its documented entry point. Configure any required model credentials or environment variables according to the project's configuration, and do not commit secrets to source control.

## Improving retrieval quality

RAG quality depends on both retrieval and generation. Useful areas to evaluate include:

- Chunk size and overlap
- PDF extraction quality and reading order
- Retrieval method and top-k passage count
- Embedding model and similarity metric, if vector retrieval is used
- Prompt structure and context length
- Handling of questions that are not answered by the document
- Grounding, source references, and citation accuracy

A fluent response is not necessarily a correct response. Evaluate whether generated claims are supported by the retrieved document passages.

## Limitations

- Scanned PDFs may require OCR.
- Tables, multi-column layouts, and complex formatting can be difficult to extract correctly.
- Retrieval may miss relevant passages or return unrelated text.
- Long documents may require careful chunking and context management.
- Model responses can contain unsupported claims unless grounding and evaluation are handled carefully.

## Privacy and security

Uploaded documents may contain confidential information. Make it clear whether files are processed locally or sent to an external model or embedding service. Avoid logging full document contents unnecessarily, protect API credentials, and apply appropriate file-size and file-type validation.

## Potential improvements

- Show page references or citations for retrieved passages.
- Add hybrid keyword and semantic retrieval.
- Support multiple PDFs and document-level filtering.
- Add conversational history with explicit context management.
- Evaluate retrieval using a representative query set.
- Add document deletion and clear retention controls.
- Detect insufficient evidence and respond accordingly.

## Project status

This README describes the RAG workflow. Update it with the actual parser, embedding model, vector store, LLM provider, UI framework, configuration variables, and startup command from the repository.






# Retrival Augument Generation - short for RAG. RAG pipeline to extact relevant documents data based on L2 or euclidean distance between prompt and document content. 




python -m venv chatbook

./chatbook/Scripts/activate 

python install -r requirements.txt

python -m uvicorn main.app:app --reload




Faiss search results

For building a RAG pipeline where you pass data retrieved from FAISS, smaller Large Language Models (LLMs) can be a good choice due to their efficiency and lower computational requirements. Here are some suggestions:
Mistral 7B: This is a powerful and efficient model from Mistral AI, known for its strong performance despite its relatively small size. It's a popular choice for various tasks and can be effectively used in RAG pipelines.
Llama 2 (7B or 13B): Meta's Llama 2 models, particularly the 7B and 13B versions, offer a good balance of size and capability. They are widely used and have a strong community backing.
Gemma (2B or 7B): Google's Gemma models are lightweight and designed for efficiency. The 2B and 7B versions are excellent for scenarios where resource constraints are a concern.
Phi-2: Microsoft's Phi-2 is another compact yet capable model, specifically trained for reasoning and language understanding. Its small size makes it suitable for deployment on less powerful hardware.
TinyLlama 1.1B: As the name suggests, TinyLlama is an extremely small model, ideal for highly constrained environments or for quick experimentation where a full-sized LLM is unnecessary.
Falcon 7B: TII's Falcon 7B is a robust model that has shown competitive performance in its size class. It can be a strong contender for your RAG pipeline.
When selecting an LLM, consider factors beyond just size, such as:
Performance on your specific task: Evaluate how well the model performs on tasks relevant to your RAG application (e.g., summarization, question answering).
Ease of integration: Check for existing integrations with your chosen framework (e.g., Hugging Face Transformers, LangChain).
Community support and resources: A larger community can provide valuable assistance and resources during development.



wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf -O models/model.gguf





https://huggingface.co/learn/llm-course/en/chapter6/3

http://localhost:8000/docs#/default/create_upload_file_uploadfile__post
