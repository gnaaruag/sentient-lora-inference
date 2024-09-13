
import pathlib
import yaml
import os
import subprocess

# GET FROM ENV
TRIGGER_WORD = os.getenv("TRIGGER_WORD")
STEPS = os.getenv("STEPS")
LOW_VRAM = os.getenv("LOW_VRAM")
SAMPLE_PROMPT = os.getenv("SAMPLE_PROMPT")
BATCH_SIZE = os.getenv("BATCH_SIZE")
SEED = os.getenv("SEED")
NAME = os.getenv("NAME")


if TRIGGER_WORD is None:
    print("TRIGGER_WORD is not set")
    TRIGGER_WORD = "tok"
    print(f"Default trigger word: {TRIGGER_WORD}")

if STEPS is None:
    print("STEPS is not set")
    STEPS = 2000
    print(f"Default steps: {STEPS}")

if LOW_VRAM is None:
    print("LOW_VRAM is not set")
    LOW_VRAM = True
    print(f"Default low_vram: {LOW_VRAM}")

if SAMPLE_PROMPT is None:
    print("SAMPLE_PROMPT is not set")
    SAMPLE_PROMPT = "a photo of a woman wearing a short sleeve round neck t-shirt full body image\n a photo of man wearing a full sleeve v neck tshirt"
    print(f"Default sample prompt: {SAMPLE_PROMPT}")
    
# seprate by \n
prompts = SAMPLE_PROMPT.split("\n")
def start_training(
    data_path: str,
    training_folder: str,
) -> None:
    print(f"trigger_word: {TRIGGER_WORD}")
 
    # get file path = ai-toolkit
    python_path = os.path.join(os.getcwd(), "ai-toolkit")
    print(f"Python path: {python_path}")
    python_file = os.path.join(python_path, "run.py")
    print(f"Python file: {python_file}")
    # yaml file = ai-toolkit/config/train_lora_flux_24gb.yaml
    yaml_file = os.path.join(os.getcwd(), "ai-toolkit", "config", "train_lora_flux_24gb.yaml")
    # training_folder = os.path.join(python_path, "output")
    print(f"Training folder: {training_folder}")
    modify_yaml(file_path="./ai-toolkit/config/train_lora_flux_24gb.yaml",
                output_file_path="./ai-toolkit/config/train_lora_flux_24gb.yaml",
                trigger_word=TRIGGER_WORD,
                folder_path=data_path,
                steps=STEPS,
                low_vram=LOW_VRAM,
                batch_size=BATCH_SIZE,
                seed=SEED,
                name=NAME,
                training_folder=training_folder,
            )
    command = [
        'conda', 'run', '-n', 'ai-toolkit-env',
        'python', python_file, yaml_file,
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Output from ai-toolkit:", process.stdout.decode('utf-8'))
    if process.stderr:
        print("Error from ai-toolkit:", process.stderr.decode('utf-8'))

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

def modify_yaml(file_path,
                output_file_path,
                name=None,
                device=None,
                lr=None,
                steps=None,
                height=None,
                width=None,
                batch_size=None,
                seed=None,
                guidance_scale=None,
                quantize=None,
                trigger_word=None,
                folder_path=None,
                low_vram=None,
                training_folder=None):
    
    # Load the existing YAML file
    data = load_yaml(file_path)

    # Modify the values if the corresponding parameters are provided
    if name:
        data['config']['name'] = name
    if device:
        data['config']['process'][0]['device'] = device
    if lr:
        data['config']['process'][0]['train']['lr'] = lr
    if steps:
        data['config']['process'][0]['train']['steps'] = steps
    if height:
        data['config']['process'][0]['sample']['height'] = height
    if width:
        data['config']['process'][0]['sample']['width'] = width
    if batch_size:
        data['config']['process'][0]['train']['batch_size'] = batch_size
    if seed:
        data['config']['process'][0]['sample']['seed'] = seed
    if guidance_scale:
        data['config']['process'][0]['sample']['guidance_scale'] = guidance_scale
    if quantize is not None:
        data['config']['process'][0]['model']['quantize'] = quantize
    if trigger_word:
        data['config']['process'][0]['trigger_word'] = trigger_word
    if folder_path:
        data['config']['process'][0]['datasets'][0]['folder_path'] = folder_path
    if low_vram is not None:
        data['config']['process'][0]['model']['low_vram'] = low_vram
    if training_folder:
        data['config']['process'][0]['training_folder'] = training_folder


    save_yaml(data, output_file_path)

