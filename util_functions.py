from typing import List, Union
import pathlib
from ntpath import join
import os
import asyncio
import zipfile
import requests


import torch

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from PIL import Image
import uuid
import zipfile
from huggingface_hub import login
from pydantic import BaseModel

from io import BytesIO





openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-2024-08-06"
OUTPUT_FOLDER = "./outputs"


class GarmentDescription(BaseModel):
    description: str
    gender: str
    garment_type: str


class GeneratePrompts(BaseModel):
    prompts: list[str]


async def generate_prompts(description, num_prompts, example_prompts, gender, SYSTEM_PROMPT_FOR_GENERATING_PROMPT):
    prompts = []
    completion = await openai.beta.chat.completions.parse(
        model=MODEL,
        response_format=GeneratePrompts,
        messages = [
            {
                "role": "system",
                # "content": f"You are a helpful assistant that constructs {num_prompts} descriptive stable diffusion prompts with various ethnicities and backgrounds. Generate prompts for selected gender: {gender} Description: {description}"
                "content" : f"{SYSTEM_PROMPT_FOR_GENERATING_PROMPT} Number of prompts: {num_prompts} Description: {description} Gender: {gender}"
            },
            {
                "role": "user",
                "content": f"Example: {(example_prompts)}",
            },
        ]
    )
    message = completion.choices[0].message
    print(message)

    if isinstance(message.parsed, GeneratePrompts):
        prompts = message.parsed.prompts

    return prompts

# # Load your images
# human_image_path = "./example/human/Jensen.jpeg"
# garment_image_path = "./example/cloth/04469_00.jpg"
# human_image = Image.open(human_image_path)
# garment_image = Image.open(garment_image_path)

# # Create the dictionary required
# input_dict = {
#     "background": human_image, 
# }

# # Define other parameters
# garment_description = "short sleeve round neck T-shirt"
# is_checked = True  # or False, depending on whether you want to auto-generate the mask
# is_checked_crop = False
# denoise_steps = 30  # Number of steps, 20-40 is typical
# seed = 42  # Random seed for reproducibility

# # Call the function
# output_image, masked_image = start_tryon(
#     dict=input_dict,
#     garm_img=garment_image,
#     garment_des=garment_description,
#     is_checked=is_checked,
#     is_checked_crop=is_checked_crop,
#     denoise_steps=denoise_steps,
#     seed=seed
# )

# # Save or display the output
# output_image.save("./output.jpg")
# masked_image.save("./masked_image.jpg")


async def generate_garment_description(image_url):
    completion = await openai.beta.chat.completions.parse(
        model=MODEL,
        response_format=GarmentDescription,
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that generates detailed descriptions of a garment with details like the color of the garment, the pattern, sleeve length, collar type, fabric, along with the gender of the person who would be wearing the garment.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate a detailed description of the garment in the image",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ],
    )
    message = completion.choices[0].message
    if isinstance(message.parsed, GarmentDescription):
        description = message.parsed.description
        gender = message.parsed.gender
        garment_type = message.parsed.garment_type
        return description, gender, garment_type


def download_image(url, folder_name, file_name):
    response = requests.get(url)
    with open(os.path.join(folder_name, file_name), "wb") as f:
        f.write(response.content)


async def crop_image_to_3by4(input_path):
    with Image.open(input_path) as img:
        width, height = img.size

        # Calculate the target width and height
        if width / height > 3 / 4:
            # Image is wider than 3:4, crop the width
            new_width = int(height * 3 / 4)
            left = (width - new_width) // 2
            right = left + new_width
            top = 0
            bottom = height
        else:
            # Image is taller than 3:4, crop the height
            new_height = int(width * 4 / 3)
            top = (height - new_height) // 2
            bottom = top + new_height
            left = 0
            right = width

        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))

        # Generate the new filename
        directory, filename = os.path.split(input_path)
        new_filename = f"cropped_{filename}"
        output_path = os.path.join(directory, new_filename)

        # Save the cropped image
        cropped_img.save(output_path)

        return output_path
    


async def overlay_image(overlay_path, base_image_path):
    # Open the base image and the overlay image
    base_image = Image.open(base_image_path).convert("RGBA")
    overlay = Image.open(overlay_path).convert("RGBA")

    # Create a new image with the size of the base image
    combined = Image.new("RGBA", base_image.size)

    # Paste the base image
    combined.paste(base_image, (0, 0))

    # Calculate position to center the overlay
    x = (base_image.width - overlay.width) // 2
    y = (base_image.height - overlay.height) // 2

    # Paste the overlay image
    combined.paste(overlay, (x, y), overlay)

    # Generate the new filename
    directory, filename = os.path.split(base_image_path)
    new_filename = f"overlayed_{filename}"
    output_path = os.path.join(directory, new_filename)

    # Save the combined image
    combined.convert("RGB").save(output_path)

    return output_path




# async def get_output_from_idm_vton(garm_img, human_img, garment_des):
#     # todo: implement
#     return "https://i.ibb.co/GRgwBxQ/cropped-flux-output-0.png"




async def zip_files(file_paths, output_zip_path):
    loop = asyncio.get_event_loop()

    async def add_file_to_zip(zip_file, file_path):
        arcname = os.path.basename(file_path)
        await loop.run_in_executor(None, zip_file.write, file_path, arcname)

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in file_paths:
            await add_file_to_zip(zip_file, file_path)

    print(f"Files zipped successfully to {output_zip_path}")





if __name__ == "__main__":
    # run flux image generation
    output_path = "outputs/image.png"
    prompt = "a photo of a woman wearing a short sleeve round neck t-shirt full body image"
    asyncio.run(get_output_from_flux(prompt, output_path))

    