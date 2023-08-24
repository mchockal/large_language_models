import json
import re
import numpy as np
from langchain.vectorstores import DeepLake

# Used to clean the inconsistencies in the format in which ChatGPT generated output.
# Also convert all characters to lower case
# Usage: clean_emotions_json("../data/spotify_song_url_emotions.json")
def clean_emotions_json(filename:str) -> None:
    with open(filename, "r") as f:
        input_data = json.load(f)

    output_data = []

    # Clean emotions data - Use only lower case letters and remove any ordered listing
    for song in input_data:
        emotions = song['emotions']
        cleaned_emotions = re.sub(r'\d+\.\s+', '', emotions.lower().replace('\n', ', '))
        output_data.append(
            {
                "song_name": song["song_name"],
                "iframe": song["iframe"],
                "emotions": cleaned_emotions
            })
        print(emotions, "\n", cleaned_emotions)

    # Write to output file which will be used to store the song emotions as embeddings
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Spotify song, url and song emotions saved to {filename}")


# Does np.random.choice and ensures we don't have duplicates in the final result
def weighted_random_sample(items: np.array, weights: np.array, n: int) -> np.array:

    indices = np.arange(len(items))
    out_indices = []

    for _ in range(n):
        chosen_index = np.random.choice(indices, p=weights)
        out_indices.append(chosen_index)

        mask = indices != chosen_index
        indices = indices[mask]
        weights = weights[mask]

        if weights.sum() != 0:
            weights = weights / weights.sum()

    return items[out_indices]

# Load DeepLake db
def load_db(dataset_path: str, *args, **kwargs) -> DeepLake:
    db = DeepLake(dataset_path, *args, **kwargs)
    return db