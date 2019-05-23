import pickle
import requests
import json
import pandas as pd
import numpy as np

from tqdm import tqdm

with open ('game_urls.pkl', 'rb') as fp:
    urls = pickle.load(fp)

# test_urls = urls[:10]

shot_types = ["SHOT", "MISSED_SHOT", "BLOCKED_SHOT", "GOAL"]

shot_df = []

for ep in tqdm(urls):

    response = requests.get(ep)
    game_json = json.loads(response.text)

    all_plays = game_json['liveData']['plays']['allPlays']

    for p in all_plays:
        # 3 subcats
        # 'about', 'result', 'coordinates'
        event_type = p['result']['eventTypeId']
        skip = False
        if event_type in shot_types:
            # print(p['coordinates'])
            if event_type == 'GOAL':
                try:
                    if p['result']['emptyNet'] == True:
                        skip = True
                except:
                    skip = False
            if p['about']['periodType'] != 'REGULAR':
                skip = True
            if not skip:
                try:
                    shot_df.append([event_type, p['coordinates']['x'], p['coordinates']['y']])
                except:
                    continue

shot_df = pd.DataFrame(shot_df, columns=['Shot Result', 'X', 'Y'])

shot_df.to_pickle("shot_locations.pkl")
