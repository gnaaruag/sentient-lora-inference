import import_hook
import shutil
import os
import asyncio

from flux import generate_image



from s3 import zip_and_upload_to_s3
from train import start_training
from util_functions import (
    generate_prompts,
    generate_garment_description,
    download_image,
    crop_image_to_3by4,
    overlay_image,
    zip_files,
)

import urllib.request 



from datetime import datetime
from PIL import Image
import requests


model_urls = {
    "densepose": {
        "model_final_162be9.pkl": "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl"
    },
    "humanparsing": {
        "parsing_atr.onnx": "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx",
        "parsing_lip.onnx": "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx"
    },
}


NUMBER_OF_OUTPUTS = 1
GARMENT_IMAGE_URL = "https://i.ibb.co/DwgvvNH/garmetnt-1.jpg"
OUTPUT_FOLDER = "./outputs"
TRIGGER_WORD = "tok"
SYSTEM_PROMPT_FOR_GENERATING_PROMPT = "You are a skilled assistant specializing in creating diverse and inclusive Stable Diffusion prompts. Your task is to generate detailed prompts that encompass a wide range of ethnicities, cultures, backgrounds, and experiences. Ensure your prompts are respectful, avoid stereotypes, and promote positive representation. Include specific details about physical appearance, clothing, setting, and context when relevant. Aim to create prompts that will result in high-quality, visually interesting images while celebrating human diversity."
STEPS = 2000
LOW_VRAM = True
SAMPLE_PROMPT = "a photo of a woman wearing a short sleeve round neck t-shirt full body image\n a photo of man wearing a full sleeve v neck tshirt"
S3_ACCESS_KEY_ID=os.getenv("S3_ACCESS_KEY_ID")
S3_ACCESS_KEY_SECRET=os.getenv("S3_ACCESS_KEY_SECRET")
S3_BUCKET=os.getenv("S3_BUCKET")
# throw error if any of the variables are not set
if os.getenv("OPENAI_API_KEY") is None:
    print("OPENAI_API_KEY is not set")
    raise ValueError("OPENAI_API_KEY is not set")

if os.getenv("S3_ACCESS_KEY_ID") is None:
    print("S3_ACCESS_KEY_ID is not set")
    raise ValueError("S3_ACCESS_KEY_ID is not set")

if os.getenv("S3_ACCESS_KEY_SECRET") is None:
    print("S3_ACCESS_KEY_SECRET is not set")
    raise ValueError("S3_ACCESS_KEY_SECRET is not set")

if os.getenv("S3_BUCKET") is None:
    print("S3_BUCKET is not set")
    raise ValueError("S3_BUCKET is not set")

if os.getenv("HF_TOKEN") is None:
    print("HF_TOKEN is not set")
    raise ValueError("HF_TOKEN is not set")

if os.getenv("GARMENT_IMAGE_URL") is None:
    print("GARMENT_IMAGE_URL is not set")
else:
    GARMENT_IMAGE_URL = os.getenv("GARMENT_IMAGE_URL")

if os.getenv("NUMBER_OF_OUTPUTS") is None:
    print("NUMBER_OF_OUTPUTS is not set")
else:
    NUMBER_OF_OUTPUTS = int(os.getenv("NUMBER_OF_OUTPUTS"))
    
if os.getenv("TRIGGER_WORD") is None:
    print("TRIGGER_WORD is not set")
else:
    TRIGGER_WORD = os.getenv("TRIGGER_WORD")
    
import shutil


def download_model(url, filepath):
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Model downloaded and saved to {filepath}")
    except Exception as e:
        print(f"Failed to download {url}: {str(e)}")

base_dir = "./ckpt"




for folder, models in model_urls.items():
    folder_path = os.path.join(base_dir, folder)
    
    # Ensure the subdirectory (and nested subfolders) exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    for model_name, model_url in models.items():
        model_path = os.path.join(folder_path, model_name)
        
        # Check if the model is already present
        if not os.path.exists(model_path):
            print(f"{model_name} not found in {folder}. Downloading...")
            download_model(model_url, model_path)
        else:
            print(f"{model_name} already exists in {folder}. Skipping download.")

from idm_vton import get_output_from_idm_vton

async def main():

    # get current date and time
    now = datetime.now()
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = os.path.join(OUTPUT_FOLDER, folder_name)
    os.makedirs(folder_name, exist_ok=True)

    # Download and save garment image
    garment_response = requests.get(GARMENT_IMAGE_URL)
    with open(f"{folder_name}/garment.jpg", "wb") as f:
        f.write(garment_response.content)

    print(f"Garment image saved to {folder_name}")

    # Step 1: Generate Description of the Garment
    garment_description, gender, garment_type = await generate_garment_description(
        GARMENT_IMAGE_URL
    )
    print(garment_description)
    print(gender)
    print(garment_type)

    sdPrompts = await generate_prompts(
        description=garment_description,
        num_prompts=NUMBER_OF_OUTPUTS,
        gender=gender,
        SYSTEM_PROMPT_FOR_GENERATING_PROMPT=SYSTEM_PROMPT_FOR_GENERATING_PROMPT,
        example_prompts=[
            f"A {'woman' if gender == 'female' else 'man'} standing, wearing a stylish floral shirt. The shirt features vibrant patterns of flowers in shades of red, yellow, and green. The person is tall, with a casual yet confident posture. They have short hair and are wearing simple jeans to complement the shirt. The background is a sunny outdoor setting, with soft natural light highlighting the floral patterns on the shirt"
        ],
    )
    print(len(sdPrompts))

    # for loop to generate outputs
    for i, prompt in enumerate(sdPrompts):
        output =  generate_image(
            prompt, os.path.join(folder_name, f"flux_output_{i}.png")
        )
        print(output)
        input_name = f"flux_output_{i}.png"
        input_path = os.path.join(folder_name, input_name)

        # Crop image to 3:4 and save with new name
        cropped_path = await crop_image_to_3by4(input_path)
        print(f"Cropped image saved as: {cropped_path}")

    # get response from idm_vton
    for i, _ in enumerate(sdPrompts):
        output = await get_output_from_idm_vton(
            f"{folder_name}/garment.jpg", f"{folder_name}/cropped_flux_output_{i}.png", garment_description, f"{folder_name}/idm_vton_output_{i}.png"
        )
        # save output to folder
        output_name = f"idm_vton_output_{i}.png"
        output_path = os.path.join(folder_name, output_name)

        # overlay original image with idm_vton output
        # overlayed_path = await overlay_image(
        #     output_path, os.path.join(folder_name, f"flux_output_{i}.png")
        # )
        print(f"Overlayed image saved as: {overlayed_path}")

    # Create a new directory for the training files
    training_folder = os.path.join(folder_name, "final_files")
    print(f"Training folder: {training_folder}")
    os.makedirs(training_folder, exist_ok=True)

    # Save the overlayed images and corresponding prompts to the training folder
    training_file_paths = [
        os.path.join(folder_name, f"idm_vton_output_{i}.png")
        for i, _ in enumerate(sdPrompts)
    ]

    for i, (file_path, prompt) in enumerate(zip(training_file_paths, sdPrompts), 1):
        # Rename the image files as 1.png, 2.png, etc.
        new_image_name = os.path.join(training_folder, f"{i}.png")
        shutil.copy(file_path, new_image_name)
        
        # Save the corresponding prompt as 1.txt, 2.txt, etc.
        with open(os.path.join(training_folder, f"{i}.txt"), "w") as txt_file:
            txt_file.write(prompt)

    print(f" files and prompts saved to: {training_folder}")
    # start training
    # give absolute path to training folder
    training_folder = os.path.join(os.getcwd(), "outputs", now.strftime("%Y-%m-%d_%H-%M-%S"), "final_files")
    print(f"Training folder: {training_folder}")
    # upload the complete folder_name to s3 via zip
    s3_key = f"uploads/{os.path.basename(folder_name)}.zip"
    zip_and_upload_to_s3(folder_name, S3_BUCKET, s3_key, S3_ACCESS_KEY_ID, S3_ACCESS_KEY_SECRET)
    # delete the folder and zip file
    shutil.rmtree(folder_name)
    os.remove(folder_name+".zip")
    


if __name__ == "__main__":
    asyncio.run(main())