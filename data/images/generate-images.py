
from tqdm import tqdm
from zipfile import ZipFile
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os, torch

# Fetch the model from the Hugging Face Hub
model_id = "stabilityai/stable-diffusion-2"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
# Move the model to the GPU
pipe.to("cuda")
# Set the scheduler to use the multistep scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Fetch the prompts
with open("/projectnb/sparkgrp/en/midjourney_prompts_filtered.txt", "r") as f:
    prompts = f.readlines()

# Generate images and save them to a zip file
with ZipFile('/projectnb/sparkgrp/en/images65k.zip','w') as zip:
    for i in tqdm(range(65000, 75000), total=10000):
        if " --" in prompts[i]:
            prompt = prompts[i].split(" --")[0]
        else:
            prompt = prompts[i]
        image = pipe(prompt, height=768, width=768).images[0]
        image.resize((500, 500))
        image.save(str(i)+'.png')
        zip.write(str(i)+'.png', arcname=str(i)+'.png')
        os.remove(str(i)+'.png')
    zip.close()
