from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, send_file
import os
import re
import shutil
import uuid
import numpy as np
import time
import subprocess
import yt_dlp
import sys
from urllib.parse import urlparse, parse_qs

from run_analysis import analyze_frame_np
from create_shorts import (
    extract_video_id,
    fetch_comments,
    extract_timestamps,
    timestamp_to_seconds,
    group_timestamps,
    download_full_video,
    create_clips_ffmpeg
)
from base64 import b64decode
from io import BytesIO
from PIL import Image

ANALYSIS_DATA = {}

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'shorts_output')
TEMP_CLIP_FOLDER = os.path.join(BASE_DIR, 'temp_clips')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(TEMP_CLIP_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PROGRESS = {}

def extract_video_id_from_url(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]
        elif parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/")[2]
    return None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/emotion_form')
def emotion_form():
    return render_template('emotion_form.html')

@app.route('/timestamp_form')
def timestamp_form():
    return render_template('timestamp_form.html')

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    youtube_url = request.form.get('youtube_url')
    if not youtube_url:
        return '유튜브 URL이 제공되지 않았습니다.', 400

    video_id = extract_video_id_from_url(youtube_url)
    if not video_id:
        return '유효하지 않은 유튜브 링크입니다.', 400

    task_id = str(uuid.uuid4())
    filename = f"{task_id}.mp4"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    ydl_opts = {
        'outtmpl': output_path,
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'merge_output_format': 'mp4'
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except Exception as e:
        return f"유튜브 다운로드 실패: {e}", 500

    return redirect(url_for('analyzing_page', youtube_id=video_id, task_id=task_id))

@app.route('/analyzing')
def analyzing_page():
    youtube_id = request.args.get('youtube_id')
    task_id = request.args.get('task_id')
    if not youtube_id or not task_id:
        return "유효하지 않은 요청입니다.", 400
    return render_template('analyzing.html', youtube_id=youtube_id, task_id=task_id)

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    data = request.get_json()
    task_id = data.get('task_id')
    if not task_id:
        return jsonify({"status": "error", "message": "task_id 누락"}), 400

    ANALYSIS_DATA[task_id] = []
    return jsonify({"status": "started"})

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    data = request.get_json()
    image_data = data.get("image")
    task_id = data.get("task_id")

    if not image_data or not task_id:
        return jsonify({"status": "error", "message": "데이터 부족"}), 400

    try:
        header, encoded = image_data.split(",", 1)
        img_bytes = b64decode(encoded)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"status": "error", "message": "이미지 디코딩 실패"}), 400

    frame = np.array(img)
    result = analyze_frame_np(frame)
    emotion = result["emotion"]
    attention = result["attention"]

    ANALYSIS_DATA.setdefault(task_id, []).append({
        "emotion": emotion,
        "attention": attention,
        "timestamp": time.time()
    })

    if emotion == "Error":
        return jsonify({"status": "error", "message": "감정 분석 실패"}), 500

    return jsonify({"status": "success", "emotion": emotion, "attention": attention})

@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    data = request.get_json()
    task_id = data.get("task_id")
    if not task_id or task_id not in ANALYSIS_DATA:
        return jsonify({"status": "error", "message": "task_id 잘못됨"}), 400

    emotion_weights = {
        "surprise": 5, "happy": 4, "sad": 3, "angry": 2, "neutral": 1
    }

    result_dir = os.path.join(RESULT_FOLDER, task_id)
    os.makedirs(result_dir, exist_ok=True)

    records = ANALYSIS_DATA[task_id]
    if not records:
        return jsonify({"status": "error", "message": "데이터 없음"}), 400

    records.sort(key=lambda x: x["timestamp"])
    min_time = records[0]["timestamp"]
    max_time = records[-1]["timestamp"]
    window_size = 10
    step = 5
    candidate_windows = []

    for t in np.arange(min_time, max_time - window_size, step):
        window = [r for r in records if t <= r["timestamp"] < t + window_size]
        if not window:
            continue

        avg_attention = sum(r["attention"] for r in window) / len(window)
        emotion_counter = {}
        for r in window:
            emotion_counter[r["emotion"]] = emotion_counter.get(r["emotion"], 0) + 1
        dominant_emotion = max(emotion_counter.items(), key=lambda x: x[1])[0]
        emotion_score = sum(emotion_weights.get(r["emotion"].lower(), 0) for r in window) / len(window)

        score = avg_attention * 0.6 + emotion_score * 0.4
        if dominant_emotion.lower() in ["error", "unknown"]:
            score *= 0.5

        candidate_windows.append({
            "start": t,
            "emotion": dominant_emotion,
            "score": score
        })

    if not candidate_windows:
        return jsonify({"status": "error", "message": "유효한 구간 없음"}), 400

    top_windows = []
    used_times = []

    for win in sorted(candidate_windows, key=lambda x: x["score"], reverse=True):
        if any(abs(win["start"] - t) < window_size for t in used_times):
            continue
        top_windows.append(win)
        used_times.append(win["start"])
        if len(top_windows) == 5:
            break

    input_video = os.path.join(UPLOAD_FOLDER, f"{task_id}.mp4")
    for idx, win in enumerate(top_windows, start=1):
        best_start = win["start"]
        best_emotion = win["emotion"]
        output_video = os.path.join(result_dir, f"short_{idx:02d}_{best_emotion}_{int(best_start)}-{int(best_start + window_size)}.mp4")

        relative_start = best_start - records[0]["timestamp"]
        if relative_start < 0:
            relative_start = 0

        safe_emotion = (best_emotion or "Neutral").capitalize().replace(":", "\\:").replace(" ", "\\ ")
        drawtext = f"drawtext=text='Emotion\\: {safe_emotion}':fontcolor=red:fontsize=24:x=10:y=H-th-10"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(relative_start),
            "-t", str(window_size),
            "-i", input_video,
            "-vf", drawtext,
            "-c:v", "libx264", "-c:a", "aac",
            output_video
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')

        if result.returncode != 0:
            return jsonify({"status": "error", "message": f"ffmpeg 실패 at short_{idx:02d}"}), 500

    return jsonify({"status": "completed"})

@app.route('/loading/<task_id>')
def loading(task_id):
    return render_template('loading.html', task_id=task_id)

@app.route('/progress/<task_id>')
def progress(task_id):
    percent = PROGRESS.get(task_id, 0)
    return jsonify({'progress': percent})

@app.route('/result/<task_id>')
def result(task_id):
    result_path = os.path.join(RESULT_FOLDER, task_id)
    if not os.path.exists(result_path):
        return "결과가 존재하지 않습니다.", 404

    files = [f for f in os.listdir(result_path) if f.endswith('.mp4')]
    videos = []
    for f in files:
        parts = f.split('_')
        emotion = parts[2].capitalize() if len(parts) > 2 else 'Neutral'
        videos.append({'filename': f"{task_id}/{f}", 'emotion': emotion})

    return render_template('result.html', videos=videos, task_id=task_id)

@app.route('/save_clip', methods=['POST'])
def save_clip():
    data = request.get_json()
    filename = data.get('filename')
    emotion = data.get('emotion')
    task_id = data.get('task_id')

    if not filename or not emotion or not task_id:
        return jsonify({'status': 'error', 'message': '잘못된 요청'}), 400

    if task_id == 'comment':
        src = os.path.join(TEMP_CLIP_FOLDER, filename)
    else:
        src = os.path.join(RESULT_FOLDER, task_id, filename)

    dest_dir = os.path.join(RESULT_FOLDER, 'categories', emotion)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)

    try:
        shutil.copyfile(src, dest)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/video/<emotion>/<filename>')
def stream_clip(emotion, filename):
    file_path = os.path.join(RESULT_FOLDER, 'categories', emotion, filename)
    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_file(file_path, mimetype='video/mp4')

    size = os.path.getsize(file_path)
    m = re.match(r'bytes=(\d+)-(\d*)', range_header)
    if not m:
        return send_file(file_path, mimetype='video/mp4')

    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else size - 1
    end = min(end, size - 1)
    length = end - start + 1

    with open(file_path, 'rb') as f:
        f.seek(start)
        data = f.read(length)

    rv = Response(data, status=206, mimetype='video/mp4', direct_passthrough=True)
    rv.headers.add('Content-Range', f'bytes {start}-{end}/{size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))
    return rv

@app.route('/categories')
def categories_view():
    emotions = ["Angry", "Happy", "Sad", "Surprise", "Neutral"]
    categories = {}
    for emo in emotions:
        dir_path = os.path.join(RESULT_FOLDER, 'categories', emo)
        clips = []
        if os.path.isdir(dir_path):
            clips = [f for f in os.listdir(dir_path) if f.endswith('.mp4')]
        categories[emo] = clips

    return render_template('categories.html', categories=categories)

@app.route('/shorts_comment', methods=['POST'])
def shorts_comment():
    youtube_url = request.form['youtube_url']
    try:
        for f in os.listdir(TEMP_CLIP_FOLDER):
            os.remove(os.path.join(TEMP_CLIP_FOLDER, f))

        result = subprocess.run(
            [sys.executable, 'create_shorts.py', youtube_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        print(result.stdout)
        print(result.stderr)

        return redirect(url_for('shorts_comment_result'))

    except Exception as e:
        return render_template('shorts_comment_result.html', error=f"에러 발생: {e}")

@app.route('/loading_page', methods=['POST'])
def loading_page():
    youtube_url = request.form['youtube_url']
    return render_template('loading.html', youtube_url=youtube_url)

@app.route('/shorts_comment_result')
def shorts_comment_result():
    result_files = sorted([f for f in os.listdir(TEMP_CLIP_FOLDER) if f.endswith('.mp4')])

    timestamp_map = {}
    timestamp_file = os.path.join(TEMP_CLIP_FOLDER, 'timestamps.txt')
    if os.path.exists(timestamp_file):
        with open(timestamp_file, 'r') as f:
            for line in f:
                fname, ts = line.strip().split(',')
                timestamp_map[fname] = int(ts)

    videos_info = []
    for f in result_files:
        ts = timestamp_map.get(f, None)
        videos_info.append({'filename': f, 'timestamps': [ts] if ts is not None else []})

    return render_template('shorts_comment_result.html', videos_info=videos_info)

@app.route('/stream/<filename>')
def stream_file(filename):
    file_path = os.path.join(TEMP_CLIP_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return "파일을 찾을 수 없습니다.", 404

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(TEMP_CLIP_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "파일을 찾을 수 없습니다.", 404

if __name__ == '__main__':
    app.run(debug=True, port=5001)
