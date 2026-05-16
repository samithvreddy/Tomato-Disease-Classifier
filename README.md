# TomatoDoc — Tomato Leaf Disease Classifier
Mini Project · 4th Semester · AI & ML

---

## What this app does

1. **Diagnose** — Upload a tomato leaf photo → get disease name, confidence score, treatment steps, and prevention tips
2. **Weather Risk** — Enter your city → 3-day disease risk forecast using real weather data (OpenWeatherMap)
3. **Soil Check** — Enter soil readings (pH, NPK, moisture) → get specific remediation advice

---

## Project structure

```
tomato-app/
├── app.py                ← Flask backend (already done)
├── disease_data.json     ← Disease info, treatments, thresholds (already done)
├── train_model.py        ← Colab training script (already done)
├── requirements.txt      ← Python dependencies
├── tomato_model.h5       ← YOU NEED TO ADD THIS (see Step 2)
├── templates/
│   └── index.html        ← Frontend HTML (already done)
├── static/
│   ├── style.css         ← Styles (already done)
│   └── app.js            ← Frontend logic (already done)
└── uploads/              ← Created automatically when app runs
```

---

## Setup instructions — do these in order

### Step 1 — Install Python dependencies

Make sure Python 3.9+ is installed. Then in a terminal:

```bash
cd tomato-app
pip install -r requirements.txt
```

This installs Flask, TensorFlow, and the other packages. TensorFlow is large (~500MB) — it will take a few minutes.

**If you get a TensorFlow error on Windows:** use `tensorflow-cpu` instead:
```bash
pip install flask werkzeug requests numpy tensorflow-cpu
```

---

### Step 2 — Train the model on Google Colab (FREE)

This is the only part that requires more time, but Colab does it for you.

1. Go to **https://colab.research.google.com** and sign in with your Google account
2. Create a new notebook
3. **Change runtime:** Runtime → Change runtime type → **T4 GPU** → Save
4. Copy the contents of `train_model.py` into Colab cells (or upload the file)
5. **Download the PlantVillage dataset from Kaggle:**
   - Go to https://www.kaggle.com/datasets/emmarex/plantdisease
   - Create a free Kaggle account if needed
   - Click Download → get `plantdisease.zip`
   - Upload this zip to Colab using the file browser (📁 icon on the left)
6. In Cell 2, uncomment the Option B lines and run
7. **Run all cells** — training takes ~20–30 minutes on Colab GPU
8. When done, download `tomato_model.h5` from the Colab file browser
9. **Place `tomato_model.h5` in the `tomato-app/` folder** (same folder as `app.py`)

> ⚠️ IMPORTANT: After training, open `class_names.json` (also downloaded from Colab) and compare with the `CLASS_NAMES` list in `app.py`. The order must match exactly. If the dataset has slightly different folder names, update the `FOLDER_TO_CLASS` dict in `train_model.py`.

---

### Step 3 — Get a free OpenWeatherMap API key

1. Go to **https://openweathermap.org/api**
2. Click "Sign Up" — it's free, no credit card needed
3. After signing in, go to **API Keys** tab
4. Copy your API key

**Set it in the app:**

Option A (recommended) — set as environment variable before running:
```bash
# On Windows (Command Prompt):
set OPENWEATHER_API_KEY=your_key_here

# On Mac/Linux:
export OPENWEATHER_API_KEY=your_key_here
```

Option B — edit `app.py` line 22 directly:
```python
WEATHER_API_KEY = 'your_key_here'
```

> The app works without a key — it runs in **demo mode** with simulated weather data. Set the key when you're ready to demo to the class.

---

### Step 4 — Run the app

```bash
cd tomato-app
python app.py
```

You should see:
```
✅ Model loaded from tomato_model.h5
 * Running on http://127.0.0.1:5000
```

Open **http://localhost:5000** in your browser. That's it.

---

### Step 5 — Test it

**For the diagnosis tab:**
- Search Google Images for "tomato early blight leaf" or "tomato late blight leaf"
- Download a clear image of just the leaf
- Upload it to the app

**For the weather tab:**
- Enter "Bengaluru" or your city name
- If the API key is set, it fetches live data; otherwise shows demo data

**For the soil tab:**
- Move the sliders to different values and click Check Soil
- Try pH 4.5 (too acidic) or high nitrogen to see warnings

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: flask` | Run `pip install -r requirements.txt` |
| `No module named tensorflow` | Run `pip install tensorflow` separately |
| `tomato_model.h5 not found` | App runs in demo mode — this is fine for now. Follow Step 2 to train. |
| Port 5000 already in use | Change `port=5000` to `port=5001` in `app.py` last line |
| Colab session times out | Save model checkpoint — Colab saves `tomato_model.h5` to disk automatically via ModelCheckpoint |
| Class order mismatch | Compare `class_names.json` (from Colab) with `CLASS_NAMES` in `app.py` — they must be in the same order |

---

## Presenting this project

**If asked "how is weather different from actual prediction?"**
> "We're not predicting weather — we're consuming real forecast data from OpenWeatherMap and applying disease-specific thresholds from agricultural research to assess risk. It's rule-based, not ML."

**If asked about confidence score:**
> "MobileNetV2 outputs a softmax probability for each class. We show that probability as the confidence score. Below 70%, we warn the user to consult an expert instead of acting on the result — which existing apps like Plantix don't do."

**If asked why only tomatoes:**
> "By focusing on one crop, we could curate accurate disease data, get a cleaner dataset, and ship something that actually works. The architecture scales — adding more crops is just retraining with more classes."

---

## References

1. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. Frontiers in Plant Science.
2. Sandler, M. et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018.
3. PlantVillage Dataset: https://www.kaggle.com/datasets/emmarex/plantdisease
4. OpenWeatherMap API: https://openweathermap.org/api
