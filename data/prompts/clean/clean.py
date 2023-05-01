import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from argparse import ArgumentParser

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--input_csv", type=str, default="diffusion_prompts.csv")
parser.add_argument("--prompt_column", type=int, default=0)
args = parser.parse_args()

# Set output csv
output_txt = args.input_csv.replace(".csv", "_filtered.txt")

# Get a list of only the prompts
df = pd.read_csv(args.input_csv)
prompts = df.iloc[:, args.prompt_column].tolist()

# Get embeddings of prompts
model = SentenceTransformer("all-MiniLM-L6-v2")

# Process data in smaller chunks
chunk_size = 1000
num_chunks = len(prompts) // chunk_size + 1
dissimilar_prompts = []

for i in tqdm(range(num_chunks), unit=" chunks"):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(prompts))
    chunk_embeddings = model.encode(prompts[start_idx:end_idx], convert_to_numpy=True, device='cuda', show_progress_bar=False)
    chunk_cosine_scores = util.pytorch_cos_sim(chunk_embeddings, chunk_embeddings)

    for j in range(chunk_cosine_scores.shape[0]):
        row_without_diagonal = np.concatenate((chunk_cosine_scores[j, :j], chunk_cosine_scores[j, j+1:]))
        dissimilar_prompts.append((row_without_diagonal < 0.9).all())

# Filter prompts
seen = set()
filtered_prompts = [
    (idx, prompt)
    for idx, (prompt, dissimilar) in enumerate(zip(prompts, dissimilar_prompts))
    if (
        # Keep only dissimilar prompts
        dissimilar
        # Remove prompts that are not strings
        and isinstance(prompt, str)
        # Remove prompts that end or start with a comma
        and not prompt.startswith(",")
        and not prompt.endswith(",")
        # Remove prompts that appear to be URLs
        and not prompt.startswith("http")
        # Remove lines that have fewer than 4 words
        and len(prompt.split()) >= 4
        # Remove identical prompts
        and prompt not in seen
        # Remove empty prompts
        and prompt != ""
    )
    and not seen.add(prompt)
]

# Save filtered prompts to a new csv, with all the original columns
df.iloc[[idx for idx, prompt in filtered_prompts], :].to_csv(args.output_csv, index=False)