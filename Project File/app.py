from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Keras model (.h5 from Hugging Face)
model = load_model("model/trash_classification_model.h5")

# Raw class names from the model
class_names = ["Plastic", "Metal", "Paper", "Cardboard", "Glass", "Trash"]

# Override rules to correct frequent misclassifications
def override_label(raw_label, filename=None):
    override_map = {
        "Metal": "Plastic",   # Metal often misused for plastic/cover
        "Paper": "Glass",     # Paper often misused for metal/glass/food
    }
    if raw_label in override_map:
        print(f"[OVERRIDE] {raw_label} → {override_map[raw_label]}")
        return override_map[raw_label]
    return raw_label

# Map corrected label into final 3-class category
def map_to_category(label):
    if label in ["Paper", "Cardboard"]:
        return "Biodegradable"
    elif label in ["Glass", "Metal", "Plastic"]:
        return "Recyclable"
    else:
        return "Trash"

# Image preprocessing to match model input
def preprocess_for_keras(img_path):
    img = Image.open(img_path).convert("RGB").resize((150, 150))  # match model input shape
    arr = np.array(img).astype(np.float32) / 255.0
    return arr.reshape(1, 150, 150, 3)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/technology')
def technology():
    return render_template('technology.html')

@app.route('/benefits')
def benefits():
    return render_template('benefits.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded image
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess & predict
    img_tensor = preprocess_for_keras(filepath)
    preds = model.predict(img_tensor)
    idx = int(np.argmax(preds[0]))
    raw_label = class_names[idx]
    confidence = round(float(preds[0][idx]) * 100, 2)

    # Apply label correction if needed
    raw_label = override_label(raw_label, filename)
    category = map_to_category(raw_label)

    return render_template(
        'result.html',
        prediction=category,
        raw_label=raw_label,
        confidence=confidence,
        filename=filename
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
