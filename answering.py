import spacy
from collections import defaultdict
from nltk.corpus import wordnet
from whoosh.index import open_dir
from whoosh.qparser import MultifieldParser
from transformers import pipeline, logging

# Print a progress bar based to show percentage completion
def progress_bar(percentage, size = 30):
    filled_blocks = int((percentage * size) // 100)
    bar = '[' + '■' * filled_blocks + '□' * (size - filled_blocks) + ']'
    output = f"\r{bar} {percentage:.2f}%"
    print(output.ljust(size + 30, ' '), end = '\r', flush = True)
    if percentage >= 100: print('\n')

# Number of synonyms to include for each search term (sometimes disabling synonyms gives better results)
NUM_SYNONYMS = 0

# Number of relevant documents to retrieve from the search results
NUM_DOCUMENTS = 5

# Number of answers to print
NUM_ANSWERS = 20

# Path to the directory where the indexed data is stored
input_folder = "indexed_data"

# The question to ask our algorithm
search_query = "What are the symptoms of COVID-19?"

# Load spaCy's NLP model and convert the search query into a spaCy doc
nlp = spacy.load('en_core_web_sm')
doc = nlp(search_query)

# Find all the entities and key terms
entities = [ent.text for ent in doc.ents]
key_terms = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN', 'VERB')]

# Combine the entities and key terms into a new search query
search_terms = set(entities + key_terms)

# Create a set of synonyms for all the search terms to help broaden Whoosh's search
synonyms = set()
for term in search_terms:
    # Keep track of synonyms for the current term
    term_synonyms = set()
    
    # Find and filter the synonyms
    for synset in wordnet.synsets(term):
        for lemma in synset.lemmas():
            term_synonyms.add(lemma.name().replace('_', ' '))
            
            # Exit out of the loops when 3 synonyms were found for the current term
            if len(term_synonyms) >= NUM_SYNONYMS: break
        if len(term_synonyms) >= NUM_SYNONYMS: break

    # Update the synonyms set with the top 3 synonyms for the term
    synonyms.update(list(term_synonyms)[:NUM_SYNONYMS])

# Add synonyms to the list of search terms amd join with Whoosh's OR operator
search_terms.update(synonyms)
search_terms = " OR ".join(list(search_terms))

# Get the indexed data
index = open_dir(input_folder)

# Tell the parser which fields to search through amd parse the search query
parser = MultifieldParser(["title", "abstract", "body"], schema = index.schema)
parsed_query = parser.parse(search_terms)

# Search and retrieve the most relevant documents
with index.searcher() as searcher:
    # Get search results
    results = searcher.search(parsed_query, limit = NUM_DOCUMENTS)
    
    # Extract and save the body paragraphs
    body_texts = []
    for result in results:
        body_texts.append(result['body'])
        
# Suppress any woarnings made by BERT     
logging.set_verbosity_error()   

# Define a pipeline with the chosen Q&A model 
qa_pipeline = pipeline("question-answering", model="dmis-lab/biobert-large-cased-v1.1-squad")

# Get answers from the Q&A model
results = []
for idx, text in enumerate(body_texts):
    paragraphs = text.split('\n')
    chunks = []
    
    # Split each paragraph into chunks of at most 512 words
    for paragraph in paragraphs:
        words = paragraph.split()
        chunks.extend([' '.join(words[i:i + 512]) for i in range(0, len(words), 512)])
    
    # Get answers for every chunk
    for jdx, chunk in enumerate(chunks):
        result = qa_pipeline(question=search_query, context=chunk)
        results.append((result['answer'], result['score']))
        
        # Print a progress bar showing how many answers generated
        progress_bar(((idx + (jdx + 1) / len(chunks)) / len(body_texts)) * 100)

# Sort the results by score in descending order
results = sorted(results, key = lambda x: x[1], reverse = True)

# Print the question
print(f"Question: {search_query}")

# Print the top answers and their scores
for answer, score in results[:NUM_ANSWERS]:
    print(f"Answer: {answer} (score: {score:0.3f})")

"""
# Calculate the best answer by summing scores of similar answers
best_results = defaultdict(float)
for answer, score in results:
    best_results[answer.lower().strip()] += score

# Choose the answer with the highest score as the best answer
best_results = sorted(best_results.items(), key=lambda item: item[1], reverse=True)

# Now you can construct the sentence
print(f"\nThe best answer is '{best_results[0][0]}' with a total score of {best_results[0][1]:0.3f}")
"""

# Choose the answer with the highest score as the best answer
print(f"\nThe best answer is '{results[0][0]}' with a score of {results[0][1]:0.3f}")