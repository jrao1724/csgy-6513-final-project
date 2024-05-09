import urllib.request
import pandas as pd
import numpy as np

sampled_vids = pd.read_csv('sampled_videos.csv')

for idx, row in sampled_vids.iterrows():
    title = row['Video Title'].replace('/', '_')
    urllib.request.urlretrieve(row['thumbnail_url'], f"thumbnails/{row['ID']}_{title}.jpg")
