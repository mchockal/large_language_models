
from dotenv import load_dotenv
load_dotenv("env_vars.env")

import os
import numpy as np
import streamlit as st
from pathlib import Path
from typing import List, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from utils import weighted_random_sample, load_db
from redis_storage import RedisStorage, UserInput

Matches = List[Tuple[Document, float]] # Custom type to save results of similarity search. Contains [(Song info, score)]
USE_STORAGE = os.environ.get("USE_STORAGE", "True").lower() in ("true", "t", "1")


@st.cache_resource
def init():
    embeddings = OpenAIEmbeddings(model=os.environ['MODEL'])
    dataset_path = f"hub://{os.environ['ACTIVELOOP_ORG_ID']}/{os.environ['DATASET']}"
    db = load_db(
        dataset_path,
        embedding=embeddings,
        token=os.environ["ACTIVELOOP_TOKEN"],
        read_only=True,
    )

    storage = RedisStorage(
        host=os.environ["UPSTASH_URL"], password=os.environ["UPSTASH_PASSWORD"]
    )

    prompt = PromptTemplate(
        input_variables=["user_input"],
        template=Path(f"{os.environ['ROOT']}/prompts/augment_user_input.prompt").read_text(),
    )

    llm = ChatOpenAI(temperature=0.3)

    chain = LLMChain(llm=llm, prompt=prompt)

    return db, chain, storage


# Don't show the setting sidebar
if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "collapsed"

st.set_page_config(initial_sidebar_state=st.session_state.sidebar_state)

# Get Vector store, LLM Chain Redis storage info. This is done once and cached.
db, chain, storage = init()

st.title("TaylorSwiftDJ 🎵🌟")
st.markdown(
    """
*<small>Made with [DeepLake](https://www.deeplake.ai/) 🚀 and [LangChain](https://python.langchain.com/en/latest/index.html) 🦜⛓️</small>*

💫 Hey there Swifties! Welcome to "TaylorSwiftDJ"! 🎤🎶 This streamlit powered app recommends her iconic songs tailored to your emotions. Dance, reminisce, and embrace your feelings with every beat – because life's a melody, and Taylor's your DJ. ✨ 🌈 💖""",
    unsafe_allow_html=True,
)
how_it_works = st.expander(label="How it works")

text_input = st.text_input(
    label="How are you feeling today?",
    placeholder="I am feeling 22!",
)

run_btn = st.button("Let's Dance! 🎶💃")
with how_it_works:
    st.markdown(
        """
The application follows a sequence of steps to deliver Taylor Swift songs matching the user's emotions:
- **User Input**: The application starts by collecting user's emotional state through a text input.
- **Emotion Encoding**: The user-provided emotions are then fed to a Language Model (LLM). The LLM interprets and encodes these emotions.
- **Similarity Search**: These encoded emotions are utilized to perform a similarity search within our [vector database](https://www.deeplake.ai/). This database houses ~130 Taylor Swift songs, each represented as emotional embeddings.
- **Song Selection**: From the pool of top matching songs, the application randomly selects one. The selection is weighted, giving preference to songs with higher similarity scores.
- **Song Retrieval**: The selected song's embedded player is displayed on the webpage for the user. Additionally, the LLM interpreted emotional state associated with the chosen song is displayed.
"""
    )


placeholder_emotions = st.empty()
placeholder = st.empty()


with st.sidebar:
    st.text("App settings")
    filter_threshold = st.slider(
        "Threshold used to filter out low scoring songs",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
    )
    max_number_of_songs = st.slider(
        "Max number of songs to get from database",
        min_value=5,
        max_value=50,
        value=20,
        step=1,
    )
    number_of_displayed_songs = st.slider(
        "Number of displayed songs", min_value=1, max_value=4, value=2, step=1
    )


def filter_scores(matches: Matches, th: float = 0.8) -> Matches:
    return [(doc, score) for (doc, score) in matches if score > th]


def normalize_scores_by_sum(matches: Matches) -> Matches:
    scores = [score for _, score in matches]
    tot = sum(scores)
    return [(doc, (score / tot)) for doc, score in matches]


def get_song(user_input: str, k: int = 20) -> Tuple[List, List]:
    emotions = chain.run(user_input=user_input)
    matches = db.similarity_search_with_score(emotions, distance_metric="cos", k=k)
    #[print(doc.metadata['name'], score) for doc, score in matches]
    scores = filter_scores(matches, filter_threshold)

    if len(scores) > 0:
        docs, scores = zip(
            *normalize_scores_by_sum(scores)
        )
        chosen_docs = weighted_random_sample(
            np.array(docs), np.array(scores), n=number_of_displayed_songs
        ).tolist()
    else:
        chosen_docs = []
    return chosen_docs, emotions


def set_song(user_input:str):
    if user_input == "":
        user_input = "I am feeling 22!"
    # take first 120 chars
    user_input = user_input[:120]
    docs, emotions = get_song(user_input, k=max_number_of_songs)
    # print(docs)
    songs = []
    spotify_iframes = []

    with placeholder_emotions:
        st.markdown("Your emotions: `" + emotions + "`")

    for doc in docs:
        name = doc.metadata["name"]
        songs.append(name)
        spotify_iframes.append(doc.metadata["iframe"])

    if len(songs) > 0:
        with placeholder:
            st.markdown("Your songs: ")
        for idx, iframe in enumerate(spotify_iframes):
            st.markdown(iframe, unsafe_allow_html=True )
    else:
        with placeholder:
            st.markdown("No song match found with the provided threshold. Please lower it. ")

    # Save user input to Redis via Upstash.
    if USE_STORAGE:
        ret_val = storage.store(
            UserInput(text=user_input, emotions=emotions, songs=songs)
        )
        if not ret_val:
            print("[ERROR] was not able to store user_input")


if run_btn:
    set_song(text_input)
