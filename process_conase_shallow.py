import os
import gzip
import pickle
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import spacy

# Target archive path on super-pc
data_path = '/home/mpguimares/conase/full_archive.tar.gz'
if not os.path.exists(data_path):
    data_path = os.path.expanduser('~/conase/full_archive.tar.gz')

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def parse_transcripts(line):
    line = line.strip()
    if not line:
        return None, None
    parts = line.split("|")
    if len(parts) != 15:
        return None, None
    try:
        word_count = float(parts[12])
        words = parts[13].split(" ")
        return words, word_count
    except (ValueError, TypeError, IndexError):
        return None, None

# Constants
CHUNK_SIZE = 10000

# Cache for lemmas
lemma_cache = {}

def get_word_lemma(word, pos):
    lemma = lemma_cache.get(word, "")
    if not lemma:
        doc = nlp(word)
        lemma = doc[0].lemma_ if len(doc) > 0 else ""
        lemma_cache[word] = lemma
    
    lemma_upper = lemma.upper()
    
    # fixes
    if lemma_upper == "'S" and pos == "VBZ":
        lemma_upper = "BE"
    elif word in ("'m", "re"):
        lemma_upper = "BE"
    elif lemma_upper in ("WEREN", "WASN"):
        lemma_upper = "BE"
    elif word == "'t":
        pos = "RP"
        lemma_upper = "NOT"
    elif word in ("'ve", "don", "didn"):
        lemma_upper = "AUX"
        
    return lemma_upper, pos

def batch_lemmatize(words_set):
    new_words = [w for w in words_set if w not in lemma_cache]
    if new_words:
        for w, doc in zip(new_words, nlp.pipe(new_words, batch_size=1000)):
            lemma_cache[w] = doc[0].lemma_ if len(doc) > 0 else ""

def count_passives(words_list):
    """Identifies passive constructions and increments passive counts directly."""
    n = len(words_list)
    for idx in range(n):
        word, pos, lemma = words_list[idx]
        if pos == "VBN" and lemma != "BE":
            # 'BE' has to come in max 5 words before the participle
            start_search = max(0, idx - 5)
            be_idx = -1
            for j in range(idx - 1, start_search - 1, -1):
                if words_list[j][2] == "BE":
                    be_idx = j
                    break
            if be_idx != -1:
                # Check if there is an intervening auxiliary between BE and the participle
                has_aux = False
                for j in range(be_idx + 1, idx):
                    if words_list[j][2] == "AUX":
                        has_aux = True
                        break
                if not has_aux:
                    bias[lemma]["passive"] += 1


vocab_size = 0
bias = defaultdict(lambda: {"active": 0, "passive": 0})

# Determine output directory
output_dir = os.path.dirname(data_path) if (data_path and os.path.exists(os.path.dirname(data_path))) else "."
bias_path = os.path.join(output_dir, "bias.pkl")
vocab_path = os.path.join(output_dir, "vocab_size.json")
csv_output_path = os.path.join(output_dir, "conase_passive_biases.csv")

def process_chunk(lines):
    global vocab_size
    parsed_chunk = []
    chunk_raw_words = set()
    
    for line in lines:
        words, count = parse_transcripts(line)
        if words is None:
            continue
        vocab_size += count
        
        line_words = []
        for raw in words:
            parts = raw.split("_", 2)
            if len(parts) >= 2:
                w = parts[0]
                p = parts[1]
            else:
                w = raw
                p = ""
            if w != "@":
                line_words.append((w, p))
                chunk_raw_words.add(w)
        parsed_chunk.append(line_words)
        
    # Batch lemmatize all unique new words in this chunk
    batch_lemmatize(chunk_raw_words)
    
    # Process each line in the chunk
    for line_words in parsed_chunk:
        processed_words = []
        for w, p in line_words:
            lemma, updated_pos = get_word_lemma(w, p)
            processed_words.append((w, updated_pos, lemma))
            
        # Update active counts
        for word, pos, lemma in processed_words:
            if pos in ("VBD", "VBP", "VBZ") and lemma != "BE":
                bias[lemma]["active"] += 1
                
        # Count shallow passives directly (no JSONL file needed)
        count_passives(processed_words)

# Main processing loop
chunk_lines = []
chunk_count = 0

with gzip.open(data_path, "rt", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing lines", unit=" lines"):
        if line.startswith("country"):
            continue
        chunk_lines.append(line)
        if len(chunk_lines) >= CHUNK_SIZE:
            process_chunk(chunk_lines)
            chunk_lines = []
            chunk_count += 1
            # Checkpoint periodically
            if chunk_count % 5 == 0:
                with open(bias_path, "wb") as f_bias:
                    pickle.dump(dict(bias), f_bias)
                with open(vocab_path, "w", encoding="utf-8") as f_vocab:
                    json.dump({"vocab_size": vocab_size}, f_vocab, indent=2)
            
    # Process remaining lines
    if chunk_lines:
        process_chunk(chunk_lines)

print(f"\nFinished processing.")
print(f"Total vocabulary size: {vocab_size}")
print(f"Total unique verbs in counts: {len(bias)}")

# Build final DataFrame with Actives, Passives, Total Occurrences, and Bias
data = []
for verb, counts in bias.items():
    act = counts["active"]
    pas = counts["passive"]
    tot = act + pas
    bias_val = round(pas / tot, 4) if tot > 0 else 0.0
    data.append({
        "verb": verb,
        "actives": act,
        "passives": pas,
        "total_occurrences": tot,
        "passive_bias": bias_val
    })

df = pd.DataFrame(data)

# Sort by total occurrences in descending order
df = df.sort_values(by="total_occurrences", ascending=False).reset_index(drop=True)
df.index += 1
df.index.name = "rank"

# Save outputs
print(f"Saving outputs to {output_dir}...")
df.to_csv(csv_output_path)
with open(bias_path, "wb") as f:
    pickle.dump(dict(bias), f)
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump({"vocab_size": vocab_size}, f, indent=2)

print(f"Successfully saved:")
print(f"  - {csv_output_path}")
print(f"  - {bias_path}")
print(f"  - {vocab_path}")

# Display the Top 25 verbs
print("\n=== TOP 25 MOST FREQUENT VERBS AND PASSIVE BIASES ===")
print(df.head(25).to_string())