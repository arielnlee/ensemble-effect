# some of the prompts are too long, and need to be truncated to 77 tokens
# the CLIP Tokenizer has a max_length of 77
# truncate all the prompts to 77 tokens

import csv
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from tqdm import tqdm

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

with open('/home/eamon/sd-clip/data/images/dataset/prompts.csv', 'r') as f:
    reader = csv.reader(f)
    with open('/home/eamon/sd-clip/data/images/dataset/prompts_truncated.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in tqdm(reader):
            prompt = row[0]
            tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=77)
            truncated_prompt = tokenizer.decode(tokenized_prompt['input_ids'][0])
            truncated_prompt = truncated_prompt.replace("<|startoftext|>", "")
            truncated_prompt = truncated_prompt.replace("<|endoftext|>", "")
            # replace row[1] when writing to csv
            row[0] = truncated_prompt
            writer.writerow(row)