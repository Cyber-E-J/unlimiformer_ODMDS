from rank_bm25 import BM25Okapi
import math

# Simulated long document, split by paragraphs
long_document = [
    "This is paragraph 1.",
    "This is paragraph 2.",
    "Here's something about dogs.",
    "Something else entirely.",
    "Cats are also great.",
    # Add more paragraphs to simulate your document
]

# Your query
query = "something about dogs and cats"

# Tokenize the paragraphs and query
tokenized_corpus = [doc.split(" ") for doc in long_document]
tokenized_query = query.split(" ")

# Initialize and train BM25 model
bm25 = BM25Okapi(tokenized_corpus)

# Get BM25 scores for the query
doc_scores = bm25.get_scores(tokenized_query)

# Sort the paragraphs by their scores in descending order
sorted_indexes = sorted(range(len(doc_scores)), key=lambda k: doc_scores[k], reverse=True)

# Extract 10% of the most relevant paragraphs
num_to_extract = math.ceil(len(long_document) * 0.1)
extracted_indexes = sorted_indexes[:num_to_extract]

# The extracted paragraphs
extracted_paragraphs = [long_document[i] for i in extracted_indexes]

# Do something with the extracted paragraphs
print("Extracted Paragraphs:")
for para in extracted_paragraphs:
    print(para)