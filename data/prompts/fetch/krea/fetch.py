# %% [markdown]
# # Fetch Prompts to Generate Images

# %% [markdown]
# ## Use krea.ai API to fetch prompts

# %%
import requests
import json
import os
import pandas as pd
from tqdm import tqdm

# %%
# Query https://devapi.krea.ai/prompts/, return is next, previous, and a list of results.
# Each result is a dictionary with keys: prompt_id, model_name, prompt, model_parameters, created_at, and generations.
# Within generations is a list of dictionaries with keys: generation_id, image_uri, thumbnail_uri, raw_data.
# Collect 500k prompt and image uri pairs, use tqdm to track progress.
# Save to data/prompts.csv

def get_prompts():
    pairs = {}
    next = 'https://devapi.krea.ai/prompts/'
    for i in tqdm(range(10000)):
        try:
            curr = len(pairs)
            r = requests.get(next)
            data = json.loads(r.text)
            for result in data['results']:
                # use lambda to get all non-empty generation uris
                pairs[result['prompt']] = list(filter(lambda x: x != '', [generation['image_uri'] for generation in result['generations']]))
            next = data['next']
        except:
            print(f'Error at {i}th iteration, {curr} pairs collected')
            print(f'Next url: {next}')
            break

    df = pd.DataFrame.from_dict(pairs, orient='index')
    df.to_csv('./prompts.csv')

get_prompts()




