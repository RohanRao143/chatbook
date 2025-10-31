import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from collections import defaultdict


def summarize_text(text, num_sentences = 3):
    try:
        stopwords_data = stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')
        stopwords_data = stopwords.words('english')

    
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)

     # Filter out stop words and punctuation
    filtered_words = [word for word in words if word.isalnum() and word not in stopwords_data]



    # Calculate word frequencies
    word_frequencies = defaultdict(int)
    for word in filtered_words:
        word_frequencies[word] += 1

    # Calculate sentence scores based on word frequencies
    sentence_scores = defaultdict(int)
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[i] += word_frequencies[word]

    # Sort sentences by score and select the top 'num_sentences'
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    # summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    
    # Reconstruct the summary in original sentence order
    summary = [sentences[i] for i in sorted(summarized_sentences)]
    
    return " ".join(summary)




# Example usage:
article_text = """
Python is a high-level, interpreted, interactive, and object-oriented scripting language. 
Python is a great language for beginner-level programmers. It is widely used in web development, 
data analysis, artificial intelligence, and scientific computing. Its simplicity and readability 
make it popular among developers. Many powerful libraries and frameworks are available for Python, 
such as Django, Flask, NumPy, Pandas, and TensorFlow.
"""

summary = summarize_text(article_text, num_sentences=2)
print(summary)





















from transformers import AutoTokenizer

from datasets import concatenate_datasets, DatasetDict


def show_samples(dataset, num_samples=3, seed=42):
    sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Title: {example['review_title']}'")
        print(f"'>> Review: {example['review_body']}'")



def filter_books(example):
    return (
        example["product_category"] == "book"
        or example["product_category"] == "digital_ebook_purchase"
    )

# spanish_books = spanish_dataset.filter(filter_books)
# english_books = english_dataset.filter(filter_books)
# show_samples(english_books)

books_dataset = books_dataset.filter(lambda x: len(x["review_title"].split()) > 2)


books_dataset = DatasetDict()

for split in english_books.keys():
    books_dataset[split] = concatenate_datasets(
        [english_books[split], spanish_books[split]]
    )
    books_dataset[split] = books_dataset[split].shuffle(seed=42)

# Peek at a few examples
show_samples(books_dataset)

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

inputs = tokenizer("I loved reading the Hunger Games!")

tokenizer.convert_ids_to_tokens(inputs.input_ids)


max_input_length = 512
max_target_length = 30


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["review_title"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = books_dataset.map(preprocess_function, batched=True)
