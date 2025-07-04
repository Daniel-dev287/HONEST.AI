import joblib
import re
import os
import pytesseract
import cv2
import numpy as np
from PIL import Image
import nltk
from flask import Flask, render_template, request, jsonify, redirect
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from datetime import datetime
import pymysql

# Koneksi ke database
def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="honest_ai",
        cursorclass=pymysql.cursors.DictCursor
    )

# Setup NLTK: download stopwords jika belum ada
nltk.download("stopwords")
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\danie\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Inisialisasi Flask
app = Flask(__name__)

# Load model klasifikasi
classifier_model = joblib.load("model_Honest_AI.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Setup library NLP
stemmer = PorterStemmer()
stop_words = set(stopwords.words("indonesian"))
spell = SpellChecker(language="en")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [spell.correction(w) if spell.correction(w) else w for w in words]
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

def classify_with_percentage(input_text):
    processed = clean_text(input_text)
    vec = vectorizer.transform([processed])
    probs = classifier_model.predict_proba(vec)[0]
    conf = probs[1] * 100
    label = "hoax" if conf < 80 else "real"
    return {"label": label, "confidence": conf}

def apply_super_resolution(image):
    return image

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)
    if gray.shape[1] < 1000:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(binary, 3)
    kernel = np.ones((1,1), np.uint8)
    morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cropped = morph[y:y+h, x:x+w]
    else:
        cropped = morph
    bordered = cv2.copyMakeBorder(cropped, 10,10,10,10, cv2.BORDER_CONSTANT, value=[255,255,255])
    return Image.fromarray(apply_super_resolution(bordered))

def detect_text_presence(image_path, area_threshold=100):
    img = preprocess_image(image_path)
    if img is None:
        return False
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if len(arr.shape)==3 else arr
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > area_threshold]
    return len(valid)>0

def extract_text_from_image_preprocessed(image_path, margin_ratio=0.05):
    img = preprocess_image(image_path)
    if img is None:
        return ""
    arr = np.array(img)
    h,w = arr.shape[:2]
    data = pytesseract.image_to_data(img, lang="eng", output_type=pytesseract.Output.DICT)
    texts = []
    for i,text in enumerate(data['text']):
        t = text.strip()
        if not t: continue
        x,y,width,height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        if x < margin_ratio*w or y < margin_ratio*h or x+width > (1-margin_ratio)*w or y+height > (1-margin_ratio)*h:
            continue
        texts.append(t)
    return " ".join(texts)

#halaman utama
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

#halaman History
@app.route("/history", methods=["GET"])
def history():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM history ORDER BY tanggal DESC LIMIT 100")
            records = cursor.fetchall()
    except Exception as e:
        print("DB Error (history):", e)
        records = []
    finally:
        conn.close()
    return render_template("history.html", records=records)

#halaman Feedback
@app.route("/feedback", methods=["GET"])
def feedback_page():
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM kotak_saran ORDER BY id DESC")
            feedback_list = cursor.fetchall()
    except Exception as e:
        print("DB Error (feedback):", e)
        feedback_list = []
    finally:
        conn.close()

    return render_template("feedback.html", feedback_list=feedback_list)

@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    message = request.form.get("message", "").strip()
    if message:
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                sql = "INSERT INTO kotak_saran (masukan, tanggal) VALUES (%s, %s)"
                val = (message, datetime.now())
                cursor.execute(sql, val)
                conn.commit()
        except Exception as e:
            print("DB Error (submit_feedback):", e)
        finally:
            conn.close()

    return redirect("/feedback")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.get("news_text", "").strip()
    if not data:
        return jsonify({"error":"Masukkan teks berita!"})
    if len(data) < 20:
        return jsonify({"error":"Teks berita terlalu pendek. Mohon tambahkan."})
    result = classify_with_percentage(data)
    # Simpan ke history
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            sql = "INSERT INTO history (tanggal, jenis_input, input_text, label, confidence) VALUES (NOW(), %s, %s, %s, %s)"
            cursor.execute(sql, ("teks", data, result['label'], result['confidence']))
            conn.commit()
    except Exception as e:
        print("DB Error (predict):", e)
    finally:
        conn.close()
    return jsonify(result)

@app.route("/predict_image", methods=["POST"])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error":"Tidak ada file gambar!"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error":"Nama file tidak valid!"}), 400
    path = os.path.join('static/uploads', file.filename)
    os.makedirs('static/uploads', exist_ok=True)
    file.save(path)
    if not detect_text_presence(path):
        os.remove(path)
        return jsonify({"error":"Gambar tidak mengandung teks."}), 400
    text = extract_text_from_image_preprocessed(path)
    if not text:
        os.remove(path)
        return jsonify({"error":"Gagal ekstrak teks."}), 400
    result = classify_with_percentage(text)
    # Simpan ke history
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            sql = "INSERT INTO history (tanggal, jenis_input, input_text, label, confidence) VALUES (NOW(), %s, %s, %s, %s)"
            cursor.execute(sql, ("gambar", text, result['label'], result['confidence']))
            conn.commit()
    except Exception as e:
        print("DB Error (predict_image):", e)
    finally:
        conn.close()
    os.remove(path)
    return jsonify({"extracted_text":text, "result":result})

if __name__ == "__main__":
    app.run(debug=True)
