import openai
from tqdm import tqdm
import os
openai.api_key = os.environ["OPENAI_API_KEY"]
import multiprocessing as mp
from tqdm import tqdm

def generate_prompt(idx):
    messages = [{"role": "system", "content": "Write a prompt for a text-to-image model."},
                {"role": "assistant", "content":"hyper realistic photo of very friendly and dystopian crater"},
                {"role": "assistant", "content":"ramen carved out of fractal rose ebony, in the style of hudson river school"},
                {"role": "assistant", "content":"ultrasaurus holding a black bean taco in the woods, near an identical cheneosaurus"},
                {"role": "assistant", "content":"a thundering retro robot crane inks on parchment with a droopy french bulldog"},
                {"role": "assistant", "content":"portrait painting of a shimmering greek hero, next to a loud frill-necked lizard"},
                {"role": "assistant", "content":"an astronaut standing on a engaging white rose, in the midst of by ivory cherry blossoms"},
                {"role": "assistant", "content":"Kaggle employee Phil at a donut shop ordering all the best donuts, with a speech bubble that proclaims 'Donuts. It's what's for dinner!'"}]

    prompt = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1
    )
            
    return prompt.choices[0].message.content


with open ("prompts.txt", "w") as f:
    with mp.Pool(mp.cpu_count()) as p:
        # Use imap_unordered to get results as they are ready and tqdm to show progress
        for prompt in tqdm(p.imap_unordered(generate_prompt, range(10000)), total=10000):
            f.write(prompt +"\n")

with open("prompts.txt", "r") as f:
    prompts = f.readlines()