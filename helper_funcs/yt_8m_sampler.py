import random
from googleapiclient.discovery import build
import pandas as pd
from pprint import pprint

# VID IDs SAMPLED FROM YOUTUBE_8M.txt FILE
# NOT COMMITED TO GITHUB DUE TO LARGE FILE SIZE

api_key = ""
youtube = build('youtube', 'v3', developerKey=api_key)

def check_availability(vid_id):
    video_info_req = youtube.videos().list(
        part='snippet,contentDetails,statistics,status',
        id=vid_id
    )

    video_info_res = video_info_req.execute()

    print(video_info_res)

    if len(video_info_res['items']) == 0:
        print("doesn't exist")
        return False, ""
    else:
        if 'regionRestriction' in video_info_res['items'][0]['contentDetails']:
            if 'blocked' in video_info_res['items'][0]['contentDetails']['regionRestriction']:
                if 'US' in video_info_res['items'][0]['contentDetails']['regionRestriction']['blocked']:
                    print("video blocked")
                    return False, ""

        print("\n\nLEGIT VIDEO; ADDING DATA")
        vid_info_dict = {
            'ID': vid_id,
            'Video Title': video_info_res['items'][0]['snippet']['title'],
            'thumbnail_url': video_info_res['items'][0]['snippet']['thumbnails']['high']['url'],
            'Views': video_info_res['items'][0]['statistics']['viewCount'] if 'viewCount' in video_info_res['items'][0]['statistics'] else "",
            'Likes': video_info_res['items'][0]['statistics']['likeCount'] if 'likeCount' in video_info_res['items'][0]['statistics'] else ""
        }
        pprint(vid_info_dict)
        return True, vid_info_dict

with open('youtube_8m_vid_ids.txt', 'r') as f:
    vid_ids = [line.strip() for line in f]

ids_len = len(vid_ids) - 1
sampled_ids = []
sampled_vids_df = pd.DataFrame()

for i in range(500):
    avail = False

    while avail is False:
        rand_choice = random.randint(0, ids_len)
        avail, data_dict = check_availability(vid_ids[rand_choice])

        if avail:
            dct = {k:[v] for k,v in data_dict.items()}
            
            sampled_vids_df = pd.concat([sampled_vids_df, pd.DataFrame(dct)])
        del vid_ids[rand_choice]
        ids_len -= 1

sampled_vids_df.to_csv('sampled_videos.csv')