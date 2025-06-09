import re
import os
import sys
import subprocess
import yt_dlp
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from collections import Counter

YOUTUBE_API_KEY = 'AIzaSyBirO4FkbsDGxAn7DIG5muexbDTkY-wLVk'
OUTPUT_DIR = 'temp_clips'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_timestamps(text):
    pattern = r'\b(?:[0-5]?\d:)?[0-5]?\d:[0-5]\d\b'
    return re.findall(pattern, text)

def timestamp_to_seconds(ts):
    parts = list(map(int, ts.split(":")))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0

def extract_video_id(url):
    parsed = urlparse(url)
    if 'youtube.com' in parsed.netloc:
        query = parse_qs(parsed.query)
        return query.get('v', [None])[0]
    elif 'youtu.be' in parsed.netloc:
        return parsed.path.lstrip('/')
    return None

def fetch_comments(video_id, max_pages=None):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    comments = []
    next_page_token = None
    page_count = 0

    while True:
        if max_pages and page_count >= max_pages:
            break

        try:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            ).execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
            page_count += 1

        except Exception:
            break

    return comments

def group_timestamps(timestamps, threshold=5):
    if not timestamps:
        return []
    grouped = []
    current_group = [timestamps[0]]

    for ts in timestamps[1:]:
        if ts - current_group[-1] <= threshold:
            current_group.append(ts)
        else:
            grouped.append(current_group)
            current_group = [ts]

    if current_group:
        grouped.append(current_group)

    return grouped

def download_full_video(url, output_path):
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'full_video.mp4')

    if os.path.exists(output_file):
        os.remove(output_file)

    ydl_opts = {
        'outtmpl': output_file,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(output_file) or os.path.getsize(output_file) < 1_000_000:
        sys.exit(1)

    return output_file

def create_clips_ffmpeg(video_path, timestamps):
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))

    timestamp_log = []

    for idx, sec in enumerate(timestamps):
        start = max(sec - 10, 0)
        duration = 20
        filename = f'short_{idx + 1}.mp4'
        short_clip_path = os.path.join(OUTPUT_DIR, filename)

        result = subprocess.run([
            'ffmpeg',
            '-ss', str(start),
            '-i', video_path,
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-y', short_clip_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')

        if result.returncode != 0:
            continue

        if os.path.getsize(short_clip_path) < 500_000:
            os.remove(short_clip_path)
            continue

        timestamp_log.append((filename, sec))

    with open(os.path.join(OUTPUT_DIR, 'timestamps.txt'), 'w') as f:
        for fname, ts in timestamp_log:
            f.write(f"{fname},{ts}\n")

def main():
    if len(sys.argv) < 2:
        return

    youtube_url = sys.argv[1]
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return

    comments = fetch_comments(video_id)
    all_timestamps = []
    for comment in comments:
        times = extract_timestamps(comment)
        all_timestamps.extend(timestamp_to_seconds(t) for t in times)

    if not all_timestamps:
        return

    timestamp_counter = Counter(all_timestamps)
    grouped = group_timestamps(sorted(timestamp_counter.keys()))

    group_reps = []
    for group in grouped:
        most_common = max(group, key=lambda t: timestamp_counter[t])
        group_reps.append((most_common, timestamp_counter[most_common]))

    top_reps = sorted(group_reps, key=lambda x: x[1], reverse=True)[:5]
    selected = [ts for ts, _ in top_reps]

    video_path = download_full_video(youtube_url, output_path='static')
    create_clips_ffmpeg(video_path, selected)

if __name__ == "__main__":
    main()
