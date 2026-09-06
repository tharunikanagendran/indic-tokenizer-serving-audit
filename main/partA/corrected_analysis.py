#!/usr/bin/env python3
"""
corrected_analysis.py — Comprehensive cross-language evaluation on 1,000 parallel sentences
(English, Hindi, Kannada, Tamil) across multiple tokenizers and multiple denominators (A3).

Evaluates:
  Tokenizers:
    - GPT-2 (tiktoken)
    - OpenAI cl100k_base (tiktoken)
    - XLM-RoBERTa (xlm-roberta-base)
    - IndicBERTv2 (ai4bharat/IndicBERTv2-MLM-only)
    - Qwen 2.5 (Qwen/Qwen2.5-7B)

  Denominators:
    - tok / word (Whitespace Word)
    - tok / grapheme (Grapheme Cluster)
    - tok / byte (UTF-8 Byte)
    - tok / sentence (Parallel Sentence)
"""

import os
import sys
import unicodedata
import regex
import pandas as pd
import tiktoken
from transformers import AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "corpus")
LANGS = ["eng", "hin", "kan", "tam"]

def count_graphemes(s: str) -> int:
    return len(regex.findall(r'\X', s))

def load_corpus():
    data = {}
    for lang in LANGS:
        fpath = os.path.join(DATA_DIR, f"{lang}.txt")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Corpus file missing: {fpath}")
        with open(fpath, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        data[lang] = lines
    return data

def load_all_tokenizers():
    print("Loading tokenizers...")
    tokenizers = {}
    
    # 1. GPT-2
    enc_gpt2 = tiktoken.get_encoding("gpt2")
    tokenizers["gpt2"] = lambda s: enc_gpt2.encode(s)

    # 2. cl100k_base (GPT-4)
    enc_cl100k = tiktoken.get_encoding("cl100k_base")
    tokenizers["cl100k_base"] = lambda s: enc_cl100k.encode(s)

    # 3. XLM-RoBERTa
    tok_xlm = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenizers["xlm-roberta-base"] = lambda s: tok_xlm.encode(s, add_special_tokens=False)

    # 4. IndicBERTv2
    tok_indic = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")
    tokenizers["indic-bert-v2"] = lambda s: tok_indic.encode(s, add_special_tokens=False)

    # 5. Qwen 2.5
    tok_qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    tokenizers["qwen2.5-7b"] = lambda s: tok_qwen.encode(s, add_special_tokens=False)

    return tokenizers

def run_evaluation():
    corpus = load_corpus()
    tokenizers = load_all_tokenizers()
    num_sentences = len(corpus["eng"])

    print(f"\nEvaluated on {num_sentences} parallel sentences per language.\n")

    # Compute corpus statistics per language
    stats = {}
    for lang in LANGS:
        lines = corpus[lang]
        tot_words = sum(len(l.split()) for l in lines)
        tot_bytes = sum(len(l.encode("utf-8")) for l in lines)
        tot_graphemes = sum(count_graphemes(l) for l in lines)
        stats[lang] = {
            "sentences": len(lines),
            "words": tot_words,
            "bytes": tot_bytes,
            "graphemes": tot_graphemes
        }

    stats_df = pd.DataFrame(stats).T
    print("--- CORPUS SUMMARY STATS ---")
    print(stats_df)
    print()

    # Results table structure
    results = []

    for tok_name, encode_fn in tokenizers.items():
        for lang in LANGS:
            lines = corpus[lang]
            tot_tokens = sum(len(encode_fn(l)) for l in lines)
            w = stats[lang]["words"]
            b = stats[lang]["bytes"]
            g = stats[lang]["graphemes"]
            s = stats[lang]["sentences"]

            tok_per_word = tot_tokens / w
            tok_per_grapheme = tot_tokens / g
            tok_per_byte = tot_tokens / b
            tok_per_sentence = tot_tokens / s

            results.append({
                "tokenizer": tok_name,
                "lang": lang,
                "tot_tokens": tot_tokens,
                "tok/word": round(tok_per_word, 3),
                "tok/grapheme": round(tok_per_grapheme, 3),
                "tok/byte": round(tok_per_byte, 3),
                "tok/sentence": round(tok_per_sentence, 2),
            })

    res_df = pd.DataFrame(results)

    print("==========================================================================")
    print("A3 FULL CORRECTED EVALUATION MATRIX (1000 PARALLEL SENTENCES)")
    print("==========================================================================")
    
    # Pivot tables for key denominators
    for denom in ["tok/word", "tok/grapheme", "tok/byte", "tok/sentence"]:
        print(f"\n--- Metric: {denom} ---")
        piv = res_df.pivot(index="tokenizer", columns="lang", values=denom)
        piv["hin/eng ratio"] = round(piv["hin"] / piv["eng"], 2)
        piv["kan/eng ratio"] = round(piv["kan"] / piv["eng"], 2)
        piv["tam/eng ratio"] = round(piv["tam"] / piv["eng"], 2)
        print(piv.to_string())

    # Save summary csv
    out_csv = os.path.join(os.path.dirname(__file__), "corrected_metrics_matrix.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved matrix to {out_csv}")

if __name__ == "__main__":
    run_evaluation()
