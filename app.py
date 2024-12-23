import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
from collections import Counter

# Flask App Setup
app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
FRAMES_FOLDER = './frames'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FRAMES_FOLDER'] = FRAMES_FOLDER
app.secret_key = "secret_key"

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

# Load the image classification pipeline
pipe = pipeline("image-classification", "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224")


def check_file_type(filename: str, allowed_extensions):
    """Check if the uploaded file is valid."""
    return filename.split('.')[-1].lower() in allowed_extensions


def analyze_image(image_path):
    """Analyze image using the pipeline."""
    try:
        image = Image.open(image_path)
        predictions = pipe(image)
        return predictions[0]  # Return top prediction
    except Exception as e:
        return f"Error analyzing image: {e}"


def extract_frames(video_path, frame_rate=1):
    """Extract frames from a video at a specific frame rate."""
    cap = cv2.VideoCapture(video_path)
    count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            frame_filename = os.path.join(app.config['FRAMES_FOLDER'], f"frame_{count}.jpg")
            cv2.imwrite(frame_filename, frame)
            frames.append(frame_filename)
        count += 1

    cap.release()
    return frames


def analyze_video(video_path):
    """Analyze video by extracting and analyzing key frames."""
    try:
        frame_paths = extract_frames(video_path, frame_rate=30)  # 1 frame per second
        predictions = []

        for frame in frame_paths:
            result = analyze_image(frame)
            if isinstance(result, dict):
                predictions.append(result['label'])

        # Get most common label
        if predictions:
            most_common_label = Counter(predictions).most_common(1)[0]
            return {"label": most_common_label[0], "count": most_common_label[1]}
        else:
            return {"error": "No frames analyzed"}
    except Exception as e:
        return f"Error analyzing video: {e}"


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Welcome to the Image and Video Upload and Analysis API",
        "routes": {
            "upload_image": "/upload/image (POST)",
            "upload_video": "/upload/video (POST)",
            "analyze_image": "/analyze_image/<filename> (GET)",
            "analyze_video": "/analyze_video/<filename> (GET)",
            "get_file": "/files/<filename> (GET)"
        }
    })


@app.route('/upload/image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if check_file_type(file.filename, ['jpg', 'jpeg', 'png', 'bmp', 'gif']):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        return jsonify({
            "message": "Image uploaded successfully",
            "filename": filename,
            "file_url": f"/files/{filename}"
        }), 200
    else:
        return jsonify({"error": "Invalid image file type."}), 400


@app.route('/upload/video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if check_file_type(file.filename, ['mp4', 'avi', 'mov']):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        return jsonify({
            "message": "Video uploaded successfully",
            "filename": filename,
            "file_url": f"/files/{filename}"
        }), 200
    else:
        return jsonify({"error": "Invalid video file type."}), 400


@app.route('/files/<filename>', methods=['GET'])
def get_file(filename):
    """Serve uploaded files."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    else:
        return jsonify({"error": "File not found"}), 404


@app.route('/analyze_image/<filename>', methods=['GET'])
def analyze_image_route(filename):
    """Analyze an uploaded image file."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        result = analyze_image(filepath)
        return jsonify({
            "filename": filename,
            "prediction": result
        }), 200
    else:
        return jsonify({"error": "File not found"}), 404


@app.route('/analyze_video/<filename>', methods=['GET'])
def analyze_video_route(filename):
    """Analyze an uploaded video file."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        result = analyze_video(filepath)
        if isinstance(result, str) and result.startswith("Error"):
            return jsonify({"error": result}), 500
        return jsonify({
            "filename": filename,
            "prediction": result
        }), 200
    else:
        return jsonify({"error": "File not found"}), 404


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
