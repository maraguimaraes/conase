import json
import pickle
from collections import Counter
import pandas as pd
from tqdm import tqdm

print("--- Step 1: Counting verb frequencies from to_analyze.jsonl ---")
verb_counter = Counter()

with open("to_analyze.jsonl", "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading fragments", unit=" lines"):
        record = json.loads(line)
        verb_counter[record["verb"]] += 1

print(f"Found {len(verb_counter):,} unique verbs across candidate passives.")

# Load active counts from bias.pkl
try:
    with open("bias.pkl", "rb") as f:
        bias = pickle.load(f)
except Exception:
    bias = {}

# Build a comprehensive frequency DataFrame
data = []
for rank, (verb, passive_cand_count) in enumerate(verb_counter.most_common(), start=1):
    active_count = bias.get(verb, {}).get("active", 0)
    data.append({
        "rank": rank,
        "verb": verb,
        "candidate_passive_count": passive_cand_count,
        "active_count": active_count,
        "total_occurrences": passive_cand_count + active_count
    })

df_all = pd.DataFrame(data)
df_all = df_all.sort_values(by="total_occurrences", ascending=False)

# Save full frequency list
df_all.to_csv("verb_frequencies_all.csv", index=False)
df_all = df_all[df_all["total_occurrences"] > 1]
print("Saved full verb frequency list to: verb_frequencies_all.csv")

# Save top 300 verbs
df_top300 = df_all.head(300)
df_top300.to_csv("top_300_verbs.csv", index=False)
print("Saved top 300 verbs to: top_300_verbs.csv")

# Step 2: Extract fragments only for the top 300 verbs
top_300_verbs_set = set(df_top300["verb"])
filtered_count = 0

print("\n--- Step 2: Creating top_300_to_analyze.jsonl for AI classification ---")
with open("to_analyze.jsonl", "r", encoding="utf-8") as infile, \
     open("top_300_to_analyze.jsonl", "w", encoding="utf-8") as outfile:
    for line in tqdm(infile, desc="Filtering top 300", unit=" lines"):
        record = json.loads(line)
        if record["verb"] in top_300_verbs_set:
            outfile.write(line)
            filtered_count += 1

print(f"Extracted {filtered_count:,} fragments for the top 300 verbs -> top_300_to_analyze.jsonl")

# Display the Top 20 verbs directly
print("\n=== TOP 20 MOST FREQUENT CANDIDATE PASSIVE VERBS ===")
print(df_top300[["rank", "verb", "candidate_passive_count", "active_count"]].head(20).to_string(index=False))