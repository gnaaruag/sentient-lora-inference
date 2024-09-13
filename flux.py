from dotenv import load_dotenv
load_dotenv()
import os
from huggingface_hub import login
import torch
from diffusers import FluxPipeline

HF_TOKEN=os.getenv("HF_TOKEN")


login(token=HF_TOKEN)
HUGGINGFACE_REPO = os.getenv("HUGGINGFACE_REPO")
LORA_NAME=os.getenv("LORA_NAME")


def initialize_pipe():
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights(HUGGINGFACE_REPO, weight_name=LORA_NAME,)
    return pipe

pipe = initialize_pipe()
TRIGGER_WORD = os.getenv("TRIGGER_WORD")


def generate_image( prompt: str, output_path: str, guidance_scale: float = 3.5, height: int = 768, width: int = 1360, num_inference_steps: int = 50):
    out = pipe(
        prompt=TRIGGER_WORD + " " + prompt,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
    ).images[0]
    out.save(output_path)
    print(f"Image saved to {output_path}")

def main():
    prompt = "a sprite on a table"
    output_path = "image.png"
    generate_image(prompt, output_path)

if __name__ == "__main__":
    main()
