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
from base64 import b64decode
from io import BytesIO
from PIL import Image

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

# 프로젝트 루트와 폴더 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # .../shortoty_web/backend
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))     # .../shortoty_web

EMOTION_UPLOAD_DIR = os.path.join(ROOT_DIR, 'emotion_uploads')
TIMESTAMP_UPLOAD_DIR = os.path.join(ROOT_DIR, 'timestamp_uploads')
SHORTS_OUTPUT_DIR = os.path.join(ROOT_DIR, 'static', 'shorts_output')
TIMESTAMP_OUTPUT_DIR = os.path.join(ROOT_DIR, 'static', 'timestamp_output')

# Flask 앱 생성 및 템플릿, static 경로 명시
TEMPLATE_DIR = os.path.join(ROOT_DIR, 'frontend', 'templates')
STATIC_DIR = os.path.join(ROOT_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

ANALYSIS_DATA = {}
PROGRESS = {}

# 필요한 폴더 미리 생성
for path in [EMOTION_UPLOAD_DIR, TIMESTAMP_UPLOAD_DIR, SHORTS_OUTPUT_DIR, TIMESTAMP_OUTPUT_DIR]:
    os.makedirs(path, exist_ok=True)


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
    output_path = os.path.join(EMOTION_UPLOAD_DIR, f"{task_id}.mp4")

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

    result_dir = os.path.join(SHORTS_OUTPUT_DIR, task_id)
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
    emotion_weights = {
        "surprise": 5, "happy": 4, "sad": 3, "angry": 2, "neutral": 1
    }

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

    input_video = os.path.join(EMOTION_UPLOAD_DIR, f"{task_id}.mp4")
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
    result_path = os.path.join(SHORTS_OUTPUT_DIR, task_id)
    if not os.path.exists(result_path):
        return "결과가 존재하지 않습니다.", 404

    files = [f for f in os.listdir(result_path) if f.endswith('.mp4')]
    videos = []
    for f in files:
        parts = f.split('_')
        emotion = parts[2].capitalize() if len(parts) > 2 else 'Neutral'
        videos.append({'filename': f"{task_id}/{f}", 'emotion': emotion})

    return render_template('result.html', videos=videos, task_id=task_id)

@app.route('/categories')
def categories_view():
    emotions = ["Angry", "Happy", "Sad", "Surprise", "Neutral"]
    categories = {}
    for emo in emotions:
        dir_path = os.path.join(SHORTS_OUTPUT_DIR, 'categories', emo)
        clips = []
        if os.path.isdir(dir_path):
            clips = [f for f in os.listdir(dir_path) if f.endswith('.mp4')]
        categories[emo] = clips

    return render_template('categories.html', categories=categories)

@app.route('/save_clip', methods=['POST'])
def save_clip():
    data = request.get_json()
    filename = data.get('filename')
    emotion = data.get('emotion')
    task_id = data.get('task_id')

    if not filename or not emotion or not task_id:
        return jsonify({'status': 'error', 'message': '잘못된 요청'}), 400

    src = os.path.join(SHORTS_OUTPUT_DIR, task_id, filename)
    dest_dir = os.path.join(SHORTS_OUTPUT_DIR, 'categories', emotion)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)

    try:
        shutil.copyfile(src, dest)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stream_clip/<emotion>/<filename>')
def stream_clip(emotion, filename):
    path = os.path.join(SHORTS_OUTPUT_DIR, 'categories', emotion, filename)
    if os.path.exists(path):
        return send_file(path)
    else:
        return "파일 없음", 404


@app.route('/stream/<task_id>/<filename>')
def stream_file(task_id, filename):
    path = os.path.join(TIMESTAMP_OUTPUT_DIR, task_id, filename)
    if os.path.exists(path):
        return send_file(path)
    else:
        return "파일 없음", 404


@app.route('/shorts_comment', methods=['POST'])
def shorts_comment():
    youtube_url = request.form['youtube_url']
    try:
        for f in os.listdir(TIMESTAMP_OUTPUT_DIR):
            path_to_file = os.path.join(TIMESTAMP_OUTPUT_DIR, f)
            if os.path.isfile(path_to_file):
                os.remove(path_to_file)

        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, 'create_shorts.py'), youtube_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        print(result.stdout)
        print(result.stderr)

        return redirect(url_for('shorts_comment_result', video_id=extract_video_id(youtube_url)))

    except Exception as e:
        return render_template('shorts_comment_result.html', error=f"에러 발생: {e}")


@app.route('/loading_page', methods=['POST'])
def loading_page():
    youtube_url = request.form['youtube_url']
    return render_template('loading.html', youtube_url=youtube_url)


@app.route('/shorts_comment_result')
def shorts_comment_result():
    video_id = request.args.get('video_id')
    folder = os.path.join(TIMESTAMP_OUTPUT_DIR, video_id)
    files = sorted([f for f in os.listdir(folder) if f.endswith('.mp4')])
    ts_path = os.path.join(folder, 'timestamps.txt')
    timestamp_map = {}
    if os.path.exists(ts_path):
        with open(ts_path) as f:
            for line in f:
                fname, ts = line.strip().split(',')
                timestamp_map[fname] = int(ts)
    videos_info = [{'filename': f"{video_id}/{f}", 'timestamps': [timestamp_map.get(f)]} for f in files]
    return render_template('shorts_comment_result.html', videos_info=videos_info)


@app.route('/download/<task_id>/<filename>')
def download_file(task_id, filename):
    file_path = os.path.join(TIMESTAMP_OUTPUT_DIR, task_id, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "파일을 찾을 수 없습니다.", 404


if __name__ == '__main__':
    app.run(debug=True, port=5001)
