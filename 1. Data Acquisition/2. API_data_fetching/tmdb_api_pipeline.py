import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('TMDB_API_KEY')
# for the first page from the api of TMDB
url = f"https://api.themoviedb.org/3/tv/top_rated?api_key={api_key}&language=en-US&page=1"


response = requests.get(url)
json_format = response.json()
json_format = response.json()['results']
df = pd.DataFrame(json_format)[['id','name','overview','first_air_date','popularity','vote_average', 'vote_count']]




# for all the pages from the api of TMDB

dfs = []

for i in range(1, 119):
    response = requests.get(
        f'https://api.themoviedb.org/3/tv/top_rated?api_key={api_key}&language=en-US&page={i}'
    )

    temp_df = pd.DataFrame(response.json()['results'])[
        ['id','name','overview','first_air_date','popularity','vote_average','vote_count']
    ]

    dfs.append(temp_df)

df = pd.concat(dfs, ignore_index=True)

df.to_csv('trending_movies.csv')
# we can also get so many APIs from RapidAPI