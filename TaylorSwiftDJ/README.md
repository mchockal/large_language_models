# TaylorSwiftDJ 🌟
This app is hosted on [HuggingFace Spaces](https://huggingface.co/spaces/mchockal/TaylorSwiftDJ)🤗
<div align = 'center'> 
    <a href ='https://taylor-swift-dj.streamlit.app/' target="_blank" rel="noopener noreferrer" >
	    <img src ='https://github.com/mchockal/large_language_models/blob/main/TaylorSwiftDJ/resources/app_screenshot.png' alt="TaylorSwiftDJ app screenshot">
    </a>
</div>

## Live Demo
- Click [HERE](https://taylor-swift-dj.streamlit.app/) for a live demo of this project hosted via Streamlit
  
## How it works
The application follows a sequence of steps to deliver Taylor Swift songs matching the user's emotions:
- **User Input**: The application starts by collecting user's emotional state through a text input.
- **Emotion Encoding**: The user-provided emotions are then fed to a Language Model (LLM). The LLM interprets and encodes these emotions.
- **Similarity Search**: These encoded emotions are utilized to perform a similarity search within our [vector database](https://www.deeplake.ai/). This database houses ~130 Taylor Swift songs, each represented as emotional embeddings.
- **Song Selection**: From the pool of top matching songs, the application randomly selects one. The selection is weighted, giving preference to songs with higher similarity scores.
- **Song Retrieval**: The selected song's embedded player is displayed on the webpage for the user. Additionally, the LLM interpreted emotional state associated with the chosen song is displayed.

## Setup instructions

### Checkout the repo
Clone the repository using the following command:
```
git clone https://huggingface.co/spaces/mchockal/TaylorSwiftDJ
```

### Set Environment Variables
You'll need to set the following variables in the `env_vars.env` file:

```dotenv
OPENAI_API_KEY=
ACTIVELOOP_TOKEN=
ACTIVELOOP_ORG_ID=
MODEL = "text-embedding-ada-002"
DATASET = <your_vector_space_for_emotions>
UPSTASH_URL= 
UPSTASH_PASSWORD=
USE_STORAGE="False"
ROOT=
```

### Install Dependencies
Install the required dependencies using the following command:
```
pip install -r requirements.txt
```

### Run the Streamlit App
Launch the Streamlit app using the following command:
```
streamlit run main.py
```

<small>_NOTE_: This project is largely inspired by [FairyTaleDJ](https://github.com/FrancescoSaverioZuppichini/FairytaleDJ)</small>
