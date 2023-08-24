
from dotenv import load_dotenv

load_dotenv(dotenv_path="../env_vars.env")

import json
from pathlib import Path
import os

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DeepLake

from utils import clean_emotions_json


"""
This function takes all the songs we have and use the lyrics to create a list of 8 emotions.
These 8 emotions will then be used for similarity matching with user prompt emotions, instead of using the entire lyrics.
"""
def generate_emotion_from_lyrics(input_file:str, prompt_path:str, output_file:str, clean_output=True) -> None:
    prompt = PromptTemplate(
        input_variables=["song_lyrics"],
        template=Path(prompt_path).read_text(),
    )
    llm = ChatOpenAI(temperature=0.8)
    chain = LLMChain(llm=llm, prompt=prompt)

    # Read file that has scraped lyrics.
    with open(input_file, "r") as f:
        lyrical_data = json.load(f)
    '''
        'song' looks like as follows
        {
            "song_name": "Cruel Summer",
            "iframe": "<iframe style=\"border-radius: 12px\" width=\"100%\" height=\"152\" title=\"Spotify Embed:...
            "lyrics": "Fever dream high in the quiet of the nightYou know that I caught it Bad... 
        }
    '''
    # Collect 8 common emotions conveyed in the songs using their lyrics
    emotion_data = []
    for song in lyrical_data:
        print(f"{song['song_name']}")
        emotions = chain.run(song_lyrics=song["lyrics"])
        emotion_data.append(
            {
                "song_name": song["song_name"],
                "iframe": song["iframe"],
                "emotions": emotions
            })
        print(emotions)

    # Write to output file which will be used to store the song emotions as embeddings
    with open(output_file, "w") as f:
        json.dump(emotion_data, f, indent=4)

    print(f"Spotify song, url and song emotions saved to {output_file}")

    # Clean the generated emotions
    if clean_output:
        clean_emotions_json("../data/spotify_song_url_emotions.json")


def create_db(dataset_path: str, input_file: str) -> DeepLake:
    with open(input_file, "r") as f:
        emotion_data = json.load(f)

    texts = []
    metadatas = []

    '''
    {
        "song_name": "Mastermind",
        "iframe": "<iframe style=\"border-radius: 12px\" width=\"100%\" height=\"152\" title=\"Spotify Embed: 
        "emotions": "excitement, happiness, love, desire, confidence, determination, vulnerability, manipulation"
    }
    '''
    for song in emotion_data:
        texts.append(song["emotions"])
        metadatas.append(
            {
                "name": song["song_name"],
                "iframe": song["iframe"],
            }
        )

    embeddings = OpenAIEmbeddings(model=os.environ['MODEL'])

    db = DeepLake.from_texts(
        texts, embeddings, metadatas=metadatas, dataset_path=dataset_path
    )

    return db

def create_emotion_embeddings(input_file:str) ->None:
    dataset_path = f"hub://{os.environ['ACTIVELOOP_ORG_ID']}/{os.environ['DATASET']}"
    create_db(dataset_path,  input_file)

if __name__ == "__main__":

    # Get top 8 emotions for each song lyric that was scraped, using GPT 3.5 Turbo
    prompt_path = "../prompts/get_song_emotions.prompt"
    input_file = "../data/spotify_song_url_lyrics.json"
    output_file = "../data/spotify_song_url_emotions.json"
    generate_emotion_from_lyrics(input_file, prompt_path, output_file, clean_output=True)

    # Convert the generated emotions to embeddings (using 'text-embedding-ada-002' model) and save them in a vector store(DeepLake)
    input_file = "../data/spotify_song_url_emotions.json"
    create_emotion_embeddings(input_file)