python -m uvicorn app:app --reload






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