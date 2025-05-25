import librosa
import numpy as np
import pandas as pd
import psycopg2
from transformers import pipeline
from deepface import DeepFace
from flask import Flask, request, jsonify
from datetime import datetime

# Database Connection
conn = psycopg2.connect("dbname=emotion_tracker user=postgres password=yourpassword")
cursor = conn.cursor()

app = Flask(__name__)

# Load text analysis model
text_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")

# 1️⃣ TEXT EMOTION ANALYSIS
def analyze_text_emotion(text):
    result = text_classifier(text)
    return result[0]['label'], result[0]['score']

# 2️⃣ FACIAL EMOTION RECOGNITION
def analyze_facial_emotion(image_path):
    analysis = DeepFace.analyze(image_path, actions=['emotion'])
    emotion = max(analysis[0]['emotion'], key=analysis[0]['emotion'].get)
    return emotion, analysis[0]['emotion'][emotion]

# 3️⃣ SPEECH EMOTION ANALYSIS
def analyze_audio_emotion(audio_path):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_mfcc = np.mean(mfcc, axis=1)
    return "Neutral", np.mean(mean_mfcc)  # Placeholder

# 4️⃣ MULTI-MODAL EMOTION FUSION
def get_final_emotion(text, image, audio):
    text_emotion, text_score = analyze_text_emotion(text)
    face_emotion, face_score = analyze_facial_emotion(image)
    audio_emotion, audio_score = analyze_audio_emotion(audio)

    # Weighted combination
    emotion_scores = {
        text_emotion: text_score * 0.4,
        face_emotion: face_score * 0.3,
        audio_emotion: audio_score * 0.3
    }

    final_emotion = max(emotion_scores, key=emotion_scores.get)
    return final_emotion

# 5️⃣ TASK RECOMMENDATION
def recommend_task(emotion):
    task_mapping = {
        "Happy": "Creative Work",
        "Stressed": "Documentation",
        "Sad": "Break or Support"
    }
    return task_mapping.get(emotion, "General Task")

# 6️⃣ STORE MOOD IN DATABASE
def store_mood(employee_id, emotion, mood_score):
    cursor.execute("INSERT INTO moods (employee_id, emotion, mood_score) VALUES (%s, %s, %s)",
                   (employee_id, emotion, mood_score))
    conn.commit()

# 7️⃣ FLASK API ROUTES
@app.route('/analyze', methods=['POST'])
def analyze_employee():
    data = request.json
    text = data['text']
    image_path = data['image']
    audio_path = data['audio']
    employee_id = data['employee_id']

    final_emotion = get_final_emotion(text, image_path, audio_path)
    task_suggestion = recommend_task(final_emotion)
    store_mood(employee_id, final_emotion, 0.5)  # Placeholder score

    return jsonify({
        "Employee ID": employee_id,
        "Emotion": final_emotion,
        "Recommended Task": task_suggestion
    })

@app.route('/dashboard', methods=['GET'])
def get_dashboard_data():
    cursor.execute("SELECT emotion, COUNT(*) FROM moods GROUP BY emotion")
    mood_counts = cursor.fetchall()
    emotion_distribution = [{"emotion": row[0], "count": row[1]} for row in mood_counts]

    return jsonify({
        "emotion_distribution": emotion_distribution
    })

if __name__ == '__main__':
    app.run(debug=True)

