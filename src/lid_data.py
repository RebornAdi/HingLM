"""
CoNLL-format loader + subword alignment for the L3Cube-HingLID task.

Each input file has one "word<tab>LANG" per line, sentences separated by
blank lines. LANG is HI or EN. We:
  1. parse into (words, word_labels) per sentence
  2. tokenize each word with the chosen tokenizer
  3. align labels to subwords: the FIRST subword of each word gets the
     word's label; all other subwords get label -100 (ignored by the loss).
"""

import os
LABEL2ID = {"HI": 0, "EN": 1}
ID2LABEL = {0: "HI", 1: "EN"}
IGNORE_INDEX = -100
def parse_conll(path):
    """Read a CoNLL file into a list of (words, labels) tuples."""
    sentences = []
    words, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if words:
                    sentences.append((words, labels))
                    words, labels = [], []
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            word = parts[0].strip()
            tag = parts[-1].strip()
            if word == "" or tag not in LABEL2ID:
                continue
            words.append(word)
            labels.append(LABEL2ID[tag])
    if words:
        sentences.append((words, labels))
    return sentences
def encode_words(words, word_labels, tokenizer, kind):
    """
    Tokenize words and align labels to subwords.
    kind = 'custom' (ByteLevelBPETokenizer) or 'gpt2' (GPT2Tokenizer).
    First subword of each word -> word label; rest -> IGNORE_INDEX.
    """
    token_ids, token_labels = [], []
    for i, (word, wlabel) in enumerate(zip(words, word_labels)):
        text = word if i == 0 else " " + word

        if kind == "custom":
            ids = tokenizer.encode(text).ids
        else:  # gpt2
            ids = tokenizer.encode(text)

        if not ids:
            continue
        token_ids.extend(ids)
        token_labels.append(wlabel)
        token_labels.extend([IGNORE_INDEX] * (len(ids) - 1))

    return token_ids, token_labels
def load_split(path, tokenizer, kind, block_size=256):
    """Parse a CoNLL file into a list of {'input_ids', 'labels'} examples."""
    sentences = parse_conll(path)
    examples = []
    for words, word_labels in sentences:
        ids, labels = encode_words(words, word_labels, tokenizer, kind)
        if not ids:
            continue
        examples.append({"input_ids": ids[:block_size], "labels": labels[:block_size]})
    return examples
if __name__ == "__main__":
    path = r"train_path"
    sents = parse_conll(path)
    print(f"Parsed {len(sents):,} sentences")
    words, labels = sents[0]
    print(f"First sentence ({len(words)} words):")
    for w, l in zip(words[:15], labels[:15]):
        print(f"  {w}\t{ID2LABEL[l]}")
