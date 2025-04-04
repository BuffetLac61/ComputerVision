#!/home/alexis/dev/bin/python3

# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText

# Vérifier CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
input("Press Enter to continue...")

tokenizer = AutoTokenizer.from_pretrained("dragonstar/image-text-captcha-v2")
model = AutoModelForImageTextToText.from_pretrained("dragonstar/image-text-captcha-v2")

model.to(device)

model.eval()
