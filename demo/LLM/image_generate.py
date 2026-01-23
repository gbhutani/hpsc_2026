import time
import torch
from diffusers import StableDiffusionPipeline
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

device = args.device
model_id = "runwayml/stable-diffusion-v1-5"

print(f"\nRunning image generation on {device.upper()}")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
pipe = pipe.to(device)

prompt = """
A futuristic supercomputer lab with glowing GPUs, cinematic lighting
"""

# Warm-up
if device == "cuda":
    _ = pipe(prompt, num_inference_steps=5)

torch.cuda.synchronize() if device == "cuda" else None

start = time.time()
image = pipe(prompt, num_inference_steps=30).images[0]
torch.cuda.synchronize() if device == "cuda" else None
end = time.time()

image.save(args.out)

print(f"Image saved to {args.out}")
print(f"Time taken: {end - start:.2f} seconds")