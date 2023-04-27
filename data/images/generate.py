# %%

# %%
from tqdm import tqdm
from zipfile import ZipFile
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os, torch

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
model_id = "stabilityai/stable-diffusion-2"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

# %%
with open("prompts.txt", "r") as f:
    prompts = f.readlines()

# %%
with ZipFile('images50k.zip','w') as zip:
    for i in tqdm(range(0, 50000), total=50000):
        image = pipe(prompts[i], height=768, width=768).images[0]
        image.resize((500, 500))
        image.save(str(i)+'.png')
        zip.write(str(i)+'.png', arcname=str(i)+'.png')
        os.remove(str(i)+'.png')
    zip.close()

with ZipFile('images100k.zip','w') as zip:
    for i in tqdm(range(50000, 100000), total=50000):
        image = pipe(prompts[i], height=768, width=768).images[0]
        image.resize((500, 500))
        image.save(str(i)+'.png')
        zip.write(str(i)+'.png', arcname=str(i)+'.png')
        os.remove(str(i)+'.png')
    zip.close()

with ZipFile('images150k.zip','w') as zip:
    for i in tqdm(range(100000, 150000), total=50000):
        image = pipe(prompts[i], height=768, width=768).images[0]
        image.resize((500, 500))
        image.save(str(i)+'.png')
        zip.write(str(i)+'.png', arcname=str(i)+'.png')
        os.remove(str(i)+'.png')
    zip.close()

with ZipFile('images200k.zip','w') as zip:
    for i in tqdm(range(150000, 200000), total=50000):
        image = pipe(prompts[i], height=768, width=768).images[0]
        image.resize((500, 500))
        image.save(str(i)+'.png')
        zip.write(str(i)+'.png', arcname=str(i)+'.png')
        os.remove(str(i)+'.png')
    zip.close()

with ZipFile('images250k.zip','w') as zip:
    for i in tqdm(range(200000, 250000), total=50000):
        image = pipe(prompts[i], height=768, width=768).images[0]
        image.resize((500, 500))
        image.save(str(i)+'.png')
        zip.write(str(i)+'.png', arcname=str(i)+'.png')
        os.remove(str(i)+'.png')
    zip.close()

with ZipFile('images300k.zip','w') as zip:
    for i in tqdm(range(250000, 300000), total=50000):
        image = pipe(prompts[i], height=768, width=768).images[0]
        image.resize((500, 500))
        image.save(str(i)+'.png')
        zip.write(str(i)+'.png', arcname=str(i)+'.png')
        os.remove(str(i)+'.png')
    zip.close()

with ZipFile('images350k.zip','w') as zip:
    for i in tqdm(range(300000, 350000), total=50000):
        image = pipe(prompts[i], height=768, width=768).images[0]
        image.resize((500, 500))
        image.save(str(i)+'.png')
        zip.write(str(i)+'.png', arcname=str(i)+'.png')
        os.remove(str(i)+'.png')
    zip.close()

with ZipFile('images400k.zip','w') as zip:
    for i in tqdm(range(350000, 400000), total=50000):
        image = pipe(prompts[i], height=768, width=768).images[0]
        image.resize((500, 500))
        image.save(str(i)+'.png')
        zip.write(str(i)+'.png', arcname=str(i)+'.png')
        os.remove(str(i)+'.png')
    zip.close()

with ZipFile('images450k.zip','w') as zip:
    for i in tqdm(range(400000, 450000), total=50000):
        image = pipe(prompts[i], height=768, width=768).images[0]
        image.resize((500, 500))
        image.save(str(i)+'.png')
        zip.write(str(i)+'.png', arcname=str(i)+'.png')
        os.remove(str(i)+'.png')
    zip.close()

with ZipFile('images500k.zip','w') as zip:
    for i in tqdm(range(450000, 500000), total=50000):
        image = pipe(prompts[i], height=768, width=768).images[0]
        image.resize((500, 500))
        image.save(str(i)+'.png')
        zip.write(str(i)+'.png', arcname=str(i)+'.png')
        os.remove(str(i)+'.png')
    zip.close()
