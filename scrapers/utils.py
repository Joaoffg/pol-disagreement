import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

def get_video_id_from_channel_info(playlist_id, max_result):
    url = 'https://youtube.googleapis.com/youtube/v3/playlistItems?part=snippet%2CcontentDetails&playlistId={}&maxResults={}&key={}'.format(playlist_id, max_result, API_KEY)

    payload, headers = {}, {}

    d = requests.request("GET", url, headers=headers, data=payload).json()

    items = d['items']
    video_ids = []

    for i in items:
        video_ids.append(i['snippet']['resourceId']['videoId'])

    return video_ids


def get_comment_threads_from_video_id(video_id, max_result):
    url = 'https://youtube.googleapis.com/youtube/v3/commentThreads?part=snippet%2Creplies&videoId={}&maxResults={}&pageToken={}&key={}'.format(video_id, max_result, '', API_KEY)
    payload, headers = {}, {}

    comment_thread = requests.request("GET", url, headers=headers, data=payload).json()

    all_items = comment_thread['items']
    while 'nextPageToken' in comment_thread:
        url = 'https://youtube.googleapis.com/youtube/v3/commentThreads?part=snippet%2Creplies&videoId={}&maxResults={}&pageToken={}&key={}'.format(video_id, max_result, comment_thread['nextPageToken'], API_KEY)
        payload, headers = {}, {}
        comment_thread = requests.request("GET", url, headers=headers, data=payload).json()
        all_items += comment_thread['items']
    
    final_form = {'type': 'comment_thread_response',
                  'video_id': video_id,
                  'total_comments': len(all_items),
                  'items': all_items,
                 }

    return final_form


def parse_comment_thread(comment_thread):
    
    for item in comment_thread['items']:
        item.pop('etag', None)
        item['snippet']['topLevelComment']['snippet'].pop('textDisplay', None)
        item['snippet']['topLevelComment']['snippet'].pop('authorDisplayName', None)
        item['snippet']['topLevelComment']['snippet'].pop('authorProfileImageUrl', None)
        item['snippet']['topLevelComment']['snippet'].pop('authorChannelUrl', None)
        item['snippet']['topLevelComment']['snippet'].pop('authorChannelId', None)
        
        if 'replies' in item:
            comments = item['replies']['comments']
            for comment in comments:
                comment.pop('etag', None)
                comment['snippet'].pop('textDisplay', None)
                comment['snippet'].pop('authorDisplayName', None)
                comment['snippet'].pop('authorProfileImageUrl', None)
                comment['snippet'].pop('authorChannelUrl', None)
                comment['snippet'].pop('authorChannelId', None)

    return comment_thread


def save_comment_threads_from_channel_uploads(channel_name, num_videos):
    video_ids = get_video_id_from_channel_info(playlist_id, num_videos)
    parent_dir = './'
    path = os.path.join(parent_dir, channel_name)

    if not os.path.isdir(path):
        os.mkdir(path)

    print('Scraping from ' + channel_name)
    count = 1
    for video_id in video_ids:
        print('Scraped {} videos'.format(count))
        
        file_name = '{}/{}.json'.format(path, video_id)
        thread = get_comment_threads_from_video_id(video_id, 500)
        with open(file_name, 'a') as f:
            json.dump(parse_comment_thread(thread), f, indent=4)

        count += 1


if __name__ == '__main__':
    pass
    
    
