import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, Response, send_file, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("crop_model.h5")
classes = ['deficiency', 'disease', 'healthy', 'pest']

camera = cv2.VideoCapture(1)  # Iriun webcam (try 0 if not working)

latest = {"health": 0, "issue": "None", "en": [], "ur": []}
last_frame = None

# ---------------- AI LOGIC ----------------
def prescription(health, issue):
    en, ur = [], []

    if health < 60:
        en.append("Increase watering")
        ur.append("پانی بڑھائیں")
    else:
        en.append("Maintain regular watering")
        ur.append("پانی معمول کے مطابق رکھیں")

    if issue == "disease":
        en.append("Spray fungicide")
        ur.append("فنگس کش اسپرے کریں")

    if issue == "pest":
        en.append("Use neem oil")
        ur.append("نیم کا تیل استعمال کریں")

    if issue == "deficiency":
        en.append("Apply fertilizer")
        ur.append("کھاد ڈالیں")

    return en, ur

def analyze_image(frame):
    img = cv2.resize(frame, (128, 128)) / 255.0
    img = img.reshape(1, 128, 128, 3)
    pred = model.predict(img, verbose=0)[0]
    issue = classes[np.argmax(pred)]
    health = int(np.max(pred) * 100)
    en, ur = prescription(health, issue)

    latest.update({
        "health": health,
        "issue": issue,
        "en": en,
        "ur": ur
    })

    save_csv(health, issue, en, ur)

def save_csv(health, issue, en, ur):
    row = {
        "Date": datetime.now(),
        "Health %": health,
        "Issue": issue,
        "English Advice": "; ".join(en),
        "Urdu Advice": "; ".join(ur)
    }
    pd.DataFrame([row]).to_csv(
        "crop_health_reports.csv",
        mode='a',
        header=not os.path.exists("crop_health_reports.csv"),
        index=False
    )

# ---------------- CAMERA STREAM ----------------
def gen_frames():
    global last_frame
    while True:
        success, frame = camera.read()
        if not success:
            break

        last_frame = frame.copy()

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template("index.html", r=latest)

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    if last_frame is not None:
        cv2.imwrite("static/captured_leaf.jpg", last_frame)
        analyze_image(last_frame)
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['leaf']
    path = "static/captured_leaf.jpg"
    file.save(path)

    frame = cv2.imread(path)
    analyze_image(frame)
    return redirect(url_for('index'))

@app.route('/download')
def download():
    return send_file("crop_health_reports.csv", as_attachment=True)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
