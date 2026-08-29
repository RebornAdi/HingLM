"""
gen_examples.py — tokenization comparison (custom vs GPT-2) for the paper.
Run from the src/ directory:  python gen_examples.py
Your custom tokenizer lives in ../tokenizer/ (vocab.json + merges.txt).
"""

from transformers import GPT2TokenizerFast

SENTENCES = [
    "Yaar aaj bohot busy hai",
    "Mujhe yeh movie pasand nahi aayi",
    "Kal party mein milte hain",
    "Thoda wait karo main aa raha hoon",
    "Yeh kitna accha idea hai yaar",
]

# --- GPT-2's own tokenizer ---
gpt2 = GPT2TokenizerFast.from_pretrained("gpt2")

# --- your custom tokenizer (vocab.json + merges.txt in ../tokenizer/) ---
custom = GPT2TokenizerFast(
    vocab_file=r"path",
    merges_file=r"path",
)

def toks(tok, s):
    ids = tok.encode(s)
    return [tok.decode([i]) for i in ids]

print("=== Tokenization comparison (custom vs GPT-2) ===\n")
for s in SENTENCES:
    c = toks(custom, s)
    g = toks(gpt2, s)
    print(f"SENTENCE: {s}   ({len(s.split())} words)")
    print(f"  Custom ({len(c)} tokens): {'|'.join(c)}")
    print(f"  GPT-2  ({len(g)} tokens): {'|'.join(g)}\n")