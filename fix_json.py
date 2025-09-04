# fix_json.py
import pandas as pd
import json

# --- CONFIGURATION ---
CSV_FILE = 'knowledge_base.csv'
EMBEDDINGS_FILE = 'data_embeddings_knowledge_base_embeddings.json'
OUTPUT_FILE = 'knowledge_base_corrected.jsonl' # We use .jsonl as the standard extension for JSON Lines

# --- SCRIPT LOGIC ---
print(f"Reading text data from: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)
texts = df['text'].tolist()

print(f"Reading embeddings from: {EMBEDDINGS_FILE}")
with open(EMBEDDINGS_FILE, 'r') as f_in, open(OUTPUT_FILE, 'w') as f_out:
    # We assume the order of embeddings in the JSON file
    # matches the order of text in the CSV file.
    for i, line in enumerate(f_in):
        if i < len(texts):
            # Load the original object with 'id' and 'embedding'
            data_obj = json.loads(line)

            # Create a new, complete object
            corrected_obj = {
                "id": data_obj.get("id", str(i)), # Use existing ID or fall back to index
                "text": texts[i], # Add the text from the CSV
                "embedding": data_obj["embedding"]
            }

            # Write the corrected object to the new file
            f_out.write(json.dumps(corrected_obj) + '\n')

print(f"\nSuccess! ✨ A new corrected file has been created: {OUTPUT_FILE}")
print("It now contains the 'id', 'text', and 'embedding' for each entry.")