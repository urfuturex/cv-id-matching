from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import fitz  # PyMuPDF for PDF handling
from werkzeug.utils import secure_filename
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json  # Import json module for handling JSON files
from datetime import datetime  # Import datetime for timestamp
from course_recommender import UdemyCourseRecommender
from PIL import Image
import pytesseract
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['HISTORY_FILE'] = 'history.json'  # Define the path for the history file
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size of 16MB

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load tokenizer and label encoder
tokenizer = joblib.load('tokenizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Load the TensorFlow model
try:
    model = load_model('model.h5')
    print("Đã tải mô hình TensorFlow thành công.")
except Exception as e:
    raise Exception(f"Đã xảy ra lỗi khi tải mô hình TensorFlow: {e}")

# Initialize history from the history file or create an empty list
def load_history():
    if os.path.exists(app.config['HISTORY_FILE']):
        try:
            with open(app.config['HISTORY_FILE'], 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Lỗi khi đọc file history.json. Khởi tạo lịch sử trống.")
            return []
    else:
        return []

def save_history():
    try:
        with open(app.config['HISTORY_FILE'], 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Đã xảy ra lỗi khi lưu lịch sử: {e}")

history = load_history()  # Load existing history

# Define prediction function
def predict(text):
    # Preprocess text
    sequences = tokenizer.texts_to_sequences([text])
    X = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    # Perform prediction
    prediction = model.predict(X)
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction, axis=1)
    # Decode label
    label = label_encoder.inverse_transform(predicted_class)
    return label[0], confidence * 100  # Returns label and confidence percentage

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'jpg', 'jpeg', 'png'}

@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an index.html template

@app.route('/predict', methods=['POST'])
def predict():
    if 'cv_file' not in request.files or 'jd_file' not in request.files:
        return redirect(request.url)
    
    cv_file = request.files['cv_file']
    jd_file = request.files['jd_file']
    
    if cv_file.filename == '' or jd_file.filename == '':
        return redirect(request.url)
    
    if cv_file and jd_file and allowed_file(cv_file.filename) and allowed_file(jd_file.filename):
        try:
            # Save files
            cv_filename = secure_filename(cv_file.filename)
            jd_filename = secure_filename(jd_file.filename)
            cv_path = os.path.join(app.config['UPLOAD_FOLDER'], cv_filename)
            jd_path = os.path.join(app.config['UPLOAD_FOLDER'], jd_filename)
            cv_file.save(cv_path)
            jd_file.save(jd_path)
            
            # Extract text using the correct function name
            cv_text = extract_text_from_pdf(cv_path)
            jd_text = extract_text_from_pdf(jd_path)
            
            # Tokenize và pad cho cả hai input
            maxlen = 100  # hoặc maxlen bạn dùng khi train
            cv_seq = tokenizer.texts_to_sequences([cv_text])
            jd_seq = tokenizer.texts_to_sequences([jd_text])
            cv_pad = pad_sequences(cv_seq, maxlen=maxlen)
            jd_pad = pad_sequences(jd_seq, maxlen=maxlen)

            # Dự đoán với model
            prediction = model.predict([cv_pad, jd_pad])
            score = float(prediction[0][0])  # Nếu output là 1 giá trị
            confidence = round(score * 100, 2)
            label = "Match" if score >= 0.5 else "Not Match"
            
            # Save to history
            entry = {
                'cv_file': cv_filename,
                'jd_file': jd_filename,
                'label': label,
                'confidence': confidence,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            history.append(entry)
            save_history()

            # === Thêm đoạn này để lấy gợi ý khóa học ===
            cv_skills = extract_skills(cv_text)
            jd_skills = extract_skills(jd_text)
            missing_skills = list(set(jd_skills) - set(cv_skills))
            course_recommendations = recommender.get_recommendations(missing_skills)
            print("Missing skills:", missing_skills)
            print("Course recommendations:", course_recommendations)
            # ==========================================

            return render_template(
                'result.html',
                label=label,
                percentage=confidence,
                cv_text=cv_text[:500],
                jd_text=jd_text[:500],
                course_recommendations=course_recommendations
            )
                                 
        except Exception as e:
            return f"Error processing files: {str(e)}", 500
    
    return redirect(request.url)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'cv' not in request.files or 'jd' not in request.files:
        return redirect(url_for('home'))
    
    cv_file = request.files['cv']
    jd_file = request.files['jd']
    
    if cv_file.filename == '' or jd_file.filename == '':
        return redirect(url_for('home'))
        
    try:
        # Save files
        cv_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(cv_file.filename))
        jd_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(jd_file.filename))
        cv_file.save(cv_path)
        jd_file.save(jd_path)
        
        # Extract text from PDFs
        cv_text = extract_text_from_pdf(cv_path)
        jd_text = extract_text_from_pdf(jd_path)

        # Tokenize và pad cho cả hai input
        maxlen = 100  # hoặc maxlen bạn dùng khi train
        cv_seq = tokenizer.texts_to_sequences([cv_text])
        jd_seq = tokenizer.texts_to_sequences([jd_text])
        cv_pad = pad_sequences(cv_seq, maxlen=maxlen)
        jd_pad = pad_sequences(jd_seq, maxlen=maxlen)

        # Dự đoán với model
        prediction = model.predict([cv_pad, jd_pad])
        score = float(prediction[0][0])  # Nếu output là 1 giá trị
        confidence = round(score * 100, 2)
        label = "Match" if score >= 0.5 else "Not Match"
        
        # Save to history
        history_entry = {
            'cv': cv_file.filename,
            'jd': jd_file.filename,
            'label': label,
            'confidence': confidence,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        history.append(history_entry)
        save_history()
        
        # Extract skills
        cv_skills = extract_skills(cv_text)
        jd_skills = extract_skills(jd_text)
        missing_skills = list(set(jd_skills) - set(cv_skills))
        course_recommendations = recommender.get_recommendations(missing_skills)
        
        return render_template(
            'result.html',
            label=label,
            percentage=confidence,
            cv_text=cv_text[:500],
            jd_text=jd_text[:500],
            course_recommendations=course_recommendations
        )
                             
    except Exception as e:
        return f"Error processing files: {str(e)}", 500

@app.route('/history')
def view_history():
    return render_template('ai.html', history=history)

# Khởi tạo recommender (test_mode=True để dùng mock data)
recommender = UdemyCourseRecommender(test_mode=True)

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    cv_file = request.files['cv_file']
    jd_file = request.files['jd_file']

    def process_file(file):
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        if ext in {'jpg', 'jpeg', 'png'}:
            # OCR ảnh, chuyển sang text
            text = image_to_text(file_path)
            # Nếu muốn, có thể tạo PDF từ text ở đây
            return text
        elif ext == 'pdf':
            return extract_text_from_pdf(file_path)
        else:
            return ""
    
    cv_text = process_file(cv_file)
    jd_text = process_file(jd_file)

    # Tokenize và pad cho cả hai input
    maxlen = 100  # hoặc maxlen bạn dùng khi train
    cv_seq = tokenizer.texts_to_sequences([cv_text])
    jd_seq = tokenizer.texts_to_sequences([jd_text])
    cv_pad = pad_sequences(cv_seq, maxlen=maxlen)
    jd_pad = pad_sequences(jd_seq, maxlen=maxlen)

    # Dự đoán với model
    prediction = model.predict([cv_pad, jd_pad])
    score = float(prediction[0][0])  # Nếu output là 1 giá trị
    confidence = round(score * 100, 2)
    label = "Match" if score >= 0.5 else "Not Match"
    
    # Save to history
    entry = {
        'cv_file': cv_file.filename,
        'jd_file': jd_file.filename,
        'label': label,
        'confidence': confidence,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    history.append(entry)
    save_history()
    
    # Extract skills
    cv_skills = extract_skills(cv_text)
    jd_skills = extract_skills(jd_text)
    missing_skills = list(set(jd_skills) - set(cv_skills))
    course_recommendations = recommender.get_recommendations(missing_skills)
    
    return render_template(
        'result.html',
        label=label,
        percentage=confidence,
        cv_text=cv_text[:500],
        jd_text=jd_text[:500],
        course_recommendations=course_recommendations
    )

def image_to_text(image_path):
    # Đọc ảnh và chuyển sang text bằng pytesseract
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='eng')
    return text

def text_to_pdf(text, pdf_path):
    # Tạo file PDF từ text
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    lines = text.split('\n')
    y = height - 40
    for line in lines:
        c.drawString(40, y, line)
        y -= 15
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_skills(text):
    skills = ["python", "java", "javascript", "sql", "aws", "docker", "react", "node.js", "html", "css", "mongodb"]
    found = []
    text = text.lower()
    for skill in skills:
        if skill in text:
            found.append(skill)
    return found

if __name__ == '__main__':
    app.run(debug=True)