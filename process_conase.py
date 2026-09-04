import os
import gzip
import pickle
import json
from collections import defaultdict
from tqdm import tqdm
import spacy

# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types

# Target archive path on super-pc
data_path = '/home/mpguimares/conase/full_archive.tar.gz'
if not os.path.exists(data_path):
    data_path = os.path.expanduser('~/conase/full_archive.tar.gz')

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Set your API key in your environment: export GOOGLE_API_KEY="your-key"
client = genai.Client()

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

def get_lemmas(word_df):
    # Keep the interface signature or remove unused functions
    pass

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

def extract_fragments(words_list):
    possible_passives = []
    n = len(words_list)
    for idx in range(n):
        word, pos, lemma = words_list[idx]
        if pos == "VBN" and lemma != "BE":
            # be has to come in max 5 words before the participle
            start_search = max(0, idx - 5)
            # Find the last "BE" in that slice
            be_idx = -1
            for j in range(idx - 1, start_search - 1, -1):
                if words_list[j][2] == "BE":
                    be_idx = j
                    break
            if be_idx != -1:
                # check if there's an auxiliary between be and the participle
                has_aux = False
                for j in range(be_idx + 1, idx):
                    if words_list[j][2] == "AUX":
                        has_aux = True
                        break
                if not has_aux:
                    start_frag = max(0, be_idx - 4)
                    end_frag = min(n, idx + 5)
                    fragment = " ".join(words_list[j][0] for j in range(start_frag, end_frag))
                    possible_passives.append(
                        {"id": idx, "verb": lemma, "fragment": fragment}
                    )
    return possible_passives


vocab_size = 0
bias = defaultdict(lambda: {"active": 0, "passive": 0})
num_candidates_found = 0

# Determine output directory
output_dir = os.path.dirname(data_path) if (data_path and os.path.exists(os.path.dirname(data_path))) else "."
to_analyze_jsonl = os.path.join(output_dir, "to_analyze.jsonl")
bias_path = os.path.join(output_dir, "bias.pkl")
vocab_path = os.path.join(output_dir, "vocab_size.json")

# Open streaming output file for fragments
out_file = open(to_analyze_jsonl, "w", encoding="utf-8")

def process_chunk(lines):
    global vocab_size, num_candidates_found
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
                
        # Extract and stream passives to disk immediately (avoids RAM explosion)
        possible_passives = extract_fragments(processed_words)
        for frag in possible_passives:
            out_file.write(json.dumps(frag) + "\n")
            num_candidates_found += 1

try:
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
                out_file.flush()
                # Checkpoint counts periodically
                if chunk_count % 5 == 0:
                    with open(bias_path, "wb") as f_bias:
                        pickle.dump(dict(bias), f_bias)
                    with open(vocab_path, "w", encoding="utf-8") as f_vocab:
                        json.dump({"vocab_size": vocab_size}, f_vocab, indent=2)
                
        # Process remaining lines
        if chunk_lines:
            process_chunk(chunk_lines)
finally:
    out_file.close()

print(f"\nFinished processing.")
print(f"Total vocabulary size: {vocab_size}")
print(f"Total candidate passive fragments: {num_candidates_found}")
print(f"Total unique verbs in active count: {len(bias)}")

print(f"Saving outputs to {output_dir}...")
with open(bias_path, "wb") as f:
    pickle.dump(dict(bias), f)

with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump({"vocab_size": vocab_size}, f, indent=2)

print(f"Successfully saved:")
print(f"  - {bias_path}")
print(f"  - {to_analyze_jsonl}")
print(f"  - {vocab_path}")