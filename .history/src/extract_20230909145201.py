from rank_bm25 import BM25Okapi
import math

# Function to count tokens in a list of paragraphs
def count_tokens(paragraphs):
    return sum(len(paragraph.split(" ")) for paragraph in paragraphs)


# Read the long document from the file
file_path = "../example_inputs/QMSum/Documents/_combined.txt"
with open(file_path, 'r') as f:
    long_document = f.readlines()


# Remove trailing whitespaces like '\n'
long_document = [line.strip() for line in long_document if line.strip()]

# Count tokens in original document
original_token_count = count_tokens(long_document)
print(f"Total tokens in original document: {original_token_count}")

# Take query input from the user
query = input("Please enter your query: ")

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


# Count tokens in extracted paragraphs
extracted_token_count = count_tokens(extracted_paragraphs)
print(f"Total tokens in extracted output: {extracted_token_count}")

# Do something with the extracted paragraphs
print("Extracted Paragraphs:")
# for para in extracted_paragraphs:
#     print(para)
