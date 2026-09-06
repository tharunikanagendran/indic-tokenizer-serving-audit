#!/usr/bin/env python3
"""
prep_corpus.py — Assembles a 1,000-sentence 4-way parallel evaluation corpus
for English (eng), Hindi (hin), Kannada (kan), and Tamil (tam).

Data source: AI4Bharat Samanantar (Parallel Corpora for Indic Languages).
Preprocessing:
  1. Filter for parallel sentences with word length between 5 and 35 words.
  2. Unicode NFC normalization.
  3. Strip leading/trailing whitespace.
"""

import os
import sys
import unicodedata
import pandas as pd
from huggingface_hub import hf_hub_download

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "corpus")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Fetching Samanantar datasets...")

    fpath_hi = hf_hub_download(repo_id='ai4bharat/samanantar', filename='hi/train-00000-of-00008.parquet', repo_type='dataset')
    fpath_kn = hf_hub_download(repo_id='ai4bharat/samanantar', filename='kn/train-00000-of-00002.parquet', repo_type='dataset')
    fpath_ta = hf_hub_download(repo_id='ai4bharat/samanantar', filename='ta/train-00000-of-00004.parquet', repo_type='dataset')

    df_hi = pd.read_parquet(fpath_hi)
    df_kn = pd.read_parquet(fpath_kn)
    df_ta = pd.read_parquet(fpath_ta)

    print("Merging on English source text...")
    merged = df_hi[['src', 'tgt']].rename(columns={'tgt': 'hin'}).merge(
        df_kn[['src', 'tgt']].rename(columns={'tgt': 'kan'}), on='src'
    ).merge(
        df_ta[['src', 'tgt']].rename(columns={'tgt': 'tam'}), on='src'
    )

    print(f"Total candidate 4-way parallel sentences: {len(merged)}")

    # Filter for quality sentence lengths (5 to 35 words in English)
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        return unicodedata.normalize("NFC", text.strip())

    filtered = []
    for idx, row in merged.iterrows():
        eng = clean_text(row['src'])
        hin = clean_text(row['hin'])
        kan = clean_text(row['kan'])
        tam = clean_text(row['tam'])

        eng_words = len(eng.split())
        hin_words = len(hin.split())
        kan_words = len(kan.split())
        tam_words = len(tam.split())

        if 5 <= eng_words <= 35 and 4 <= hin_words <= 40 and 3 <= kan_words <= 35 and 3 <= tam_words <= 35:
            filtered.append({'eng': eng, 'hin': hin, 'kan': kan, 'tam': tam})
            if len(filtered) >= 1000:
                break

    print(f"Selected {len(filtered)} clean parallel sentence tuples.")

    for lang in ['eng', 'hin', 'kan', 'tam']:
        out_path = os.path.join(OUTPUT_DIR, f"{lang}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in filtered:
                f.write(item[lang] + "\n")
        print(f"Wrote {out_path} ({len(filtered)} lines)")

if __name__ == "__main__":
    main()
