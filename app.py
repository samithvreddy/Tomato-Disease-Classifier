import os
import json
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max

# ── Load disease data ──────────────────────────────────────────
with open('disease_data.json') as f:
    DISEASE_DATA = json.load(f)

# ── Class names — must match the order your model was trained on ──
# PlantVillage tomato classes (alphabetical order from Kaggle dataset)
CLASS_NAMES = [
    'Tomato_Bacterial_Spot',
    'Tomato_Early_Blight',
    'Tomato_Late_Blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_Leaf_Spot',
    'Tomato_Spider_Mites',
    'Tomato_Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_healthy',
    'Tomato_mosaic_virus',
]
# ── Load model ─────────────────────────────────────────────────
model = None

def load_model():
    global model
    try:
        import tensorflow as tf
        if os.path.exists('tomato_model.h5'):
            model = tf.keras.models.load_model('tomato_model.h5')
            print("✅ Model loaded from tomato_model.h5")
        else:
            print("⚠️  tomato_model.h5 not found — running in demo mode")
    except ImportError:
        print("⚠️  TensorFlow not installed — running in demo mode")

def predict_disease(image_path):
    """Run inference. Returns (class_name, confidence_float)."""
    if model is None:
        # Demo mode: return a random class so the UI still works
        import random
        cls = random.choice(CLASS_NAMES)
        conf = round(random.uniform(0.55, 0.97), 4)
        return cls, conf

    try:
        import tensorflow as tf
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        arr = tf.keras.preprocessing.image.img_to_array(img)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        preds = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))
        return CLASS_NAMES[idx], float(preds[idx])
    except Exception as e:
        print(f"Inference error: {e}")
        return 'Tomato_healthy', 0.5

# ── Soil advice lookup ─────────────────────────────────────────
TOMATO_SOIL_IDEALS = {
    'ph':       {'min': 6.0, 'max': 7.0},
    'moisture': {'min': 60,  'max': 80},   # % field capacity
    'nitrogen': {'min': 2,   'max': 5},    # level 1-5 scale
    'phosphorus':{'min': 2,  'max': 4},
    'potassium': {'min': 2,  'max': 4},
}

def soil_advice(ph, moisture, nitrogen, phosphorus, potassium):
    issues = []
    advice = []

    def check(name, val, lo, hi, low_msg, high_msg):
        if val < lo:
            issues.append({'param': name, 'status': 'low', 'value': val, 'ideal': f'{lo}–{hi}'})
            advice.append(low_msg)
        elif val > hi:
            issues.append({'param': name, 'status': 'high', 'value': val, 'ideal': f'{lo}–{hi}'})
            advice.append(high_msg)
        else:
            issues.append({'param': name, 'status': 'ok', 'value': val, 'ideal': f'{lo}–{hi}'})

    check('pH', ph, 6.0, 7.0,
          'pH is too acidic — apply agricultural lime (1–2 kg per 10 sq m) to raise pH.',
          'pH is too alkaline — add sulphur powder or organic matter (compost) to lower pH.')

    check('Moisture', moisture, 60, 80,
          'Soil is too dry — increase irrigation frequency or use drip irrigation.',
          'Soil is too wet — reduce irrigation and improve drainage to avoid root rot.')

    check('Nitrogen (N)', nitrogen, 2, 5,
          'Low nitrogen — apply urea (46-0-0) at 5g/plant or use vermicompost.',
          'Excess nitrogen — reduce fertiliser. Excess N causes lush leaves but poor fruiting and increases blight risk.')

    check('Phosphorus (P)', phosphorus, 2, 4,
          'Low phosphorus — apply single super phosphate (SSP) at transplanting.',
          'Excess phosphorus — stop P application. Excess blocks zinc and iron uptake.')

    check('Potassium (K)', potassium, 2, 4,
          'Low potassium — apply muriate of potash (MOP) or sulphate of potash at 10g/plant.',
          'Excess potassium — reduce K fertiliser. Excess K blocks magnesium uptake.')

    ok_count = sum(1 for i in issues if i['status'] == 'ok')
    overall = 'good' if ok_count == 5 else ('fair' if ok_count >= 3 else 'poor')

    return {'overall': overall, 'issues': issues, 'advice': advice}

# ── Weather risk engine ────────────────────────────────────────
WEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', 'YOUR_API_KEY_HERE')

def get_weather_risk(city):
    """Fetch 3-day forecast and compute disease risk per day."""
    if WEATHER_API_KEY == 'YOUR_API_KEY_HERE':
        # Demo data so the UI works without a real key
        return _demo_weather(city)

    try:
        url = f'https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_API_KEY}&units=metric&cnt=24'
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {'error': f'Could not fetch weather: {str(e)}'}

    # Group forecast items by day
    from collections import defaultdict
    from datetime import datetime
    days = defaultdict(list)
    for item in data.get('list', []):
        day = datetime.fromtimestamp(item['dt']).strftime('%A, %d %b')
        days[day].append({
            'temp': item['main']['temp'],
            'humidity': item['main']['humidity'],
        })

    result = []
    for day, readings in list(days.items())[:3]:
        avg_temp = round(sum(r['temp'] for r in readings) / len(readings), 1)
        avg_hum  = round(sum(r['humidity'] for r in readings) / len(readings), 1)
        risk, risk_diseases = compute_risk(avg_temp, avg_hum)
        result.append({
            'day': day,
            'temp': avg_temp,
            'humidity': avg_hum,
            'risk': risk,
            'risk_diseases': risk_diseases,
        })

    return {'city': city, 'forecast': result}

def compute_risk(temp, humidity):
    """Rule-based disease risk from temp + humidity."""
    at_risk = []
    for cls, data in DISEASE_DATA.items():
        if cls == 'Tomato_healthy':
            continue
        wt = data['weather_triggers']
        if humidity >= wt['humidity_threshold'] and wt['temp_min'] <= temp <= wt['temp_max']:
            at_risk.append(data['display'])

    if len(at_risk) >= 3:
        return 'high', at_risk
    elif len(at_risk) >= 1:
        return 'medium', at_risk
    else:
        return 'low', []

def _demo_weather(city):
    """Returns plausible demo weather for UI testing."""
    import random
    days_names = ['Monday, 12 May', 'Tuesday, 13 May', 'Wednesday, 14 May']
    forecast = []
    for day in days_names:
        temp = round(random.uniform(22, 32), 1)
        humidity = round(random.uniform(65, 88), 1)
        risk, diseases = compute_risk(temp, humidity)
        forecast.append({'day': day, 'temp': temp, 'humidity': humidity,
                         'risk': risk, 'risk_diseases': diseases})
    return {'city': city, 'forecast': forecast, 'demo': True}

# ── Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed = {'jpg', 'jpeg', 'png', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': 'Only JPG, PNG, and WEBP images are supported'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(save_path)

    class_name, confidence = predict_disease(save_path)
    disease = DISEASE_DATA.get(class_name, DISEASE_DATA['Tomato_healthy'])

    return jsonify({
        'class': class_name,
        'confidence': round(confidence * 100, 1),
        'low_confidence': confidence < 0.70,
        'disease': disease,
    })

@app.route('/weather', methods=['GET'])
def weather():
    city = request.args.get('city', '').strip()
    if not city:
        return jsonify({'error': 'City name is required'}), 400
    return jsonify(get_weather_risk(city))

@app.route('/soil', methods=['POST'])
def soil():
    data = request.get_json()
    try:
        ph         = float(data.get('ph', 6.5))
        moisture   = float(data.get('moisture', 70))
        nitrogen   = float(data.get('nitrogen', 3))
        phosphorus = float(data.get('phosphorus', 3))
        potassium  = float(data.get('potassium', 3))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid values provided'}), 400

    result = soil_advice(ph, moisture, nitrogen, phosphorus, potassium)
    return jsonify(result)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)
