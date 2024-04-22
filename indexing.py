import os
import json
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID

# Print a progress bar based to show percentage completion
def progress_bar(percentage, size = 30):
    filled_blocks = int((percentage * size) // 100)
    bar = '[' + '■' * filled_blocks + '□' * (size - filled_blocks) + ']'
    output = f"\r{bar} {percentage:.2f}%"
    print(output.ljust(size + 30, ' '), end = '\r', flush = True)
    if percentage >= 100: print('\n')

# Number of documents to index
NUM_DOCUMENTS = 10000

# Path to the directory where the JSON files are stored
INPUT_FOLDER = "C:/Users/eddyi/Downloads/COVID-19-research/document_parses/pdf_json"

# Define output folder for indexing data or create one if it doesn't exist
output_folder = "indexed_data"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    
# Define which information will be indexed and stored
schema = Schema(title = TEXT(stored = True),
                abstract = TEXT(stored = True),
                body = TEXT(stored = True),
                paper_id = ID(stored = True))

# Create the index
index = create_in(output_folder, schema)

# Open a writer for the index
writer = index.writer()

# Get all the documents from the input folder
files = os.listdir(INPUT_FOLDER)[:NUM_DOCUMENTS]

# Index the documents
for idx, file in enumerate(files):
    with open(os.path.join(INPUT_FOLDER, file), 'r', encoding='utf-8') as f:
        # Load a document
        doc = json.load(f)
        
        # Extract and join the paper ID, title, abstract, and body
        paper_id = doc['paper_id']
        title = doc['metadata'].get('title', 'No Title')
        abstract = "\n".join([ab['text'] for ab in doc.get('abstract', []) if ab.get('text')])
        body =     "\n".join([bt['text'] for bt in doc.get('body_text', []) if bt.get('text')])
        
        # Add the extracted information to the index
        writer.add_document(title = title, abstract = abstract, body = body, paper_id = paper_id)
        
        progress_bar(((idx + 1) / len(files)) * 100)

# Commit changes to the index
print("Committing changes to index...\n")
writer.commit()

print(f"Finished indexing {NUM_DOCUMENTS} documents!")