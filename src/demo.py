"""
Minimal demo UI for your trained model. Run after you have at least one checkpoint.

    python demo.py --checkpoint ../checkpoints/rope_step5000.pt --tokenizer_dir ../tokenizer

Opens a local Gradio interface; add `share=True` in launch() if you want a
public link to put in your README / show in an interview.
"""

import argparse

import gradio as gr
import torch
from tokenizers import ByteLevelBPETokenizer

from config import ModelConfig
from model import GPT


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = GPT(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, mcfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_dir", default="../tokenizer")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mcfg = load_model(args.checkpoint, device)
    tokenizer = ByteLevelBPETokenizer(
        f"{args.tokenizer_dir}/vocab.json", f"{args.tokenizer_dir}/merges.txt"
    )

    def generate(prompt, max_new_tokens, temperature, top_k):
        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, int(max_new_tokens), temperature=temperature, top_k=int(top_k))
        return tokenizer.decode(out[0].tolist())

    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(label="Prompt", value="Yaar aaj mausam"),
            gr.Slider(10, 200, value=80, step=10, label="Max new tokens"),
            gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature"),
            gr.Slider(1, 100, value=40, step=1, label="Top-k"),
        ],
        outputs=gr.Textbox(label="Generated text"),
        title="Hinglish Small LM — pretrained from scratch",
        description=f"{model.num_params()/1e6:.1f}M param transformer, trained from random init on a custom Hindi-English code-mixed corpus.",
    )
    demo.launch()


if __name__ == "__main__":
    main()
