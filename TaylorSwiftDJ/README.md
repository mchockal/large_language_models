# TaylorSwiftDJ 🎵🌟

*<small>Made with [DeepLake](https://www.deeplake.ai/) 🚀 and [LangChain](https://python.langchain.com/en/latest/index.html) 🦜⛓️</small>*

💫 Hey there Swifties! Welcome to "TaylorSwiftDJ"! 🎤🎶 This app recommends Taylor Swift's iconic songs tailored to your emotions. Dance, reminisce, and embrace your feelings with every beat – because life's a melody, and Taylor's your DJ. ✨ 🌈 💖

## Live Demo
- Click [HERE]() for a live demo of this project hosted via Streamlit
  
## How it works
The application follows a sequence of steps to deliver Taylor Swift songs matching the user's emotions:
- **User Input**: The application starts by collecting user's emotional state through a text input.
- **Emotion Encoding**: The user-provided emotions are then fed to a Language Model (LLM). The LLM interprets and encodes these emotions.
- **Similarity Search**: These encoded emotions are utilized to perform a similarity search within our [vector database](https://www.deeplake.ai/). This database houses ~130 Taylor Swift songs, each represented as emotional embeddings.
- **Song Selection**: From the pool of top matching songs, the application randomly selects one. The selection is weighted, giving preference to songs with higher similarity scores.
- **Song Retrieval**: The selected song's embedded player is displayed on the webpage for the user. Additionally, the LLM interpreted emotional state associated with the chosen song is displayed.

## Setup instructions
--TODO--

<small>_NOTE_: This project is largely inspired by [FairyTaleDJ](https://github.com/FrancescoSaverioZuppichini/FairytaleDJ)</small>
