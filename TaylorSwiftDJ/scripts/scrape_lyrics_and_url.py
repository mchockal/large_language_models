import os

import requests
import json
import re

from bs4 import BeautifulSoup

'''
Get album names and id's
Get all track id's from each album id's
'''

def get_spotify_creds():
    # Replace with your Spotify API credentials
    CLIENT_ID = "f0035b10765a4cfebb434b857cf41300"
    CLIENT_SECRET = "845e93b8a8284586bb6491d3f94be685"

    # Set up the authentication headers
    auth_url = "https://accounts.spotify.com/api/token"
    auth_data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    auth_response = requests.post(auth_url, data=auth_data)
    auth_response_data = auth_response.json()
    access_token = auth_response_data["access_token"]

    # Set up the API request headers
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    return headers


def get_genius_creds():
    # Replace with your Spotify API credentials
    CLIENT_ID = "r_RcNRjkfYwoqNF3lmaIqw1y4T09Z5XIVjftPLdygQJiCEoBfFNA7oXe6gqF4q6m"
    CLIENT_SECRET = "W7v6s2Ka_y_4CsrZk-2pNcfXPzCWSZkzrNanXI2jLDjU8tr00ABfEIdgfHqjWGKAFapBdwMNumLc0vu9veZlvw"

    # Set up the authentication headers
    auth_url = "https://api.genius.com/oauth/token"
    auth_data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    auth_response = requests.post(auth_url, data=auth_data)
    auth_response_data = auth_response.json()
    access_token = auth_response_data["access_token"]

    # Set up the API request headers
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    return headers

def get_iframe(track_id:str) ->str:
    api_url = "https://open.spotify.com/oembed?url=https%3A%2F%2Fopen.spotify.com%2Ftrack%2F"+str(track_id)
    response = requests.get(api_url, headers=spotify_headers)
    data = response.json()
    return data['html']

def get_genius_url(track_name, artist="Taylor-swift"):
    track_name = track_name.replace(" ", "-").lower()
    track_name = re.sub(r'[^a-zA-Z0-9\s-]', '', track_name)
    api_url = f"https://genius.com/{artist}-{track_name}-lyrics"
    return api_url

def scrape_lyrics(track_name:str, artist="Taylor-swift") -> str:
    api_url = get_genius_url(track_name, artist)
    response = requests.get(api_url)
    if response.status_code == 404:
        return None
    html = BeautifulSoup(response.text, 'html.parser')
    lyrics = html.find('div', class_='Lyrics__Container-sc-1ynbvzw-5 Dzxov').get_text()
    # remove identifiers like chorus, verse, etc
    lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
    # remove empty lines
    lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])
    return lyrics

if __name__ == "__main__":
    # Get all songs from 'This is Taylor Swift' playlist.
    # The API returns only 50 songs per request - so we loop till we get them all
    # Once we have song name, scrape song lyrics from genius-lyrics.
    # TODO: Use asyncio for get_iframe() and scrape_lyrics

    offset = 0
    flag = True
    spotify_headers = get_spotify_creds()
    genius_headers = get_genius_creds()
    scraped_songs = []

    while flag:
        api_url = "https://api.spotify.com/v1/playlists/37i9dQZF1DX5KpP2LN299J/tracks?market=US&fields=items%28track%28name%2C+id%29%29&limit=50&offset="+str(offset)
        response = requests.get(api_url, headers=spotify_headers)
        data = response.json()
        songs = data["items"]
        count = 0
        for idx, song in enumerate(songs):
            count += 1
            song_name = song["track"]["name"]
            song_id = song["track"]["id"]
            iframe = get_iframe(song_id)
            lyrics = scrape_lyrics(song_name)
            if lyrics is None:
                continue
            print("Song Name:", song_name)
            print("Iframe: ", iframe)
            print("Lyrics: ", lyrics[0:50])
            print("-" * 30)
            song_details = {
                "song_name": song_name,
                "iframe": iframe,
                "lyrics": lyrics
            }
            scraped_songs.append(song_details)
        if count < 50 :
            flag = False
        offset += 50

    # Save the scraped song data to a JSON file
    output_file = "../data/spotify_song_url_lyrics.json"
    with open(output_file, "w") as f:
        json.dump(scraped_songs, f, indent=4)

    print(f"Scraped song data saved to {output_file}")


