"""
Flask Web Application – Emotion Recognition Demo
=================================================
Affective Computing Homework

Routes:
  GET  /              → main UI page
  POST /api/predict   → JSON prediction endpoint
  GET  /api/health    → model status check
"""

import os
import re
import string

import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

try:
    from flask_cors import CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
except Exception:
    # Keep app runnable even if flask-cors is not installed yet
    pass

# ── Emotion meta ─────────────────────────────────────────────────────────────
EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Valence/Arousal from Russell's (1980) Circumplex Model of Affect
EMOTION_CONFIG = {
    'sadness':  {'emoji': '😢', 'color': '#60A5FA', 'valence': -0.6, 'arousal': -0.5},
    'joy':      {'emoji': '😊', 'color': '#FBBF24', 'valence':  0.8, 'arousal':  0.6},
    'love':     {'emoji': '❤️', 'color': '#F472B6', 'valence':  0.7, 'arousal':  0.2},
    'anger':    {'emoji': '😠', 'color': '#F87171', 'valence': -0.7, 'arousal':  0.8},
    'fear':     {'emoji': '😨', 'color': '#A78BFA', 'valence': -0.5, 'arousal':  0.7},
    'surprise': {'emoji': '😲', 'color': '#34D399', 'valence':  0.3, 'arousal':  0.9},
}

MODEL_PATH = 'model/emotion_classifier.pkl'
pipeline = None  # loaded at startup


def patch_legacy_pipeline(model_pipeline):
    """Patch known sklearn compatibility fields for older/newer pickles."""
    try:
        clf = model_pipeline.named_steps.get('clf')
        if clf is not None and not hasattr(clf, 'multi_class'):
            clf.multi_class = 'auto'
    except Exception:
        pass
    return model_pipeline


# ── Text preprocessing (must match train.py) ─────────────────────────────────
def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Feature importance: top single-words driving the prediction ──────────────
def get_top_words(text_clean: str, predicted_idx: int, n: int = 6):
    """Return the n highest-scoring single words for the predicted emotion."""
    try:
        tfidf = pipeline.named_steps['tfidf']
        clf   = pipeline.named_steps['clf']

        text_vec     = tfidf.transform([text_clean])
        feature_names = tfidf.get_feature_names_out()
        coef          = clf.coef_[predicted_idx]

        nonzero = text_vec.nonzero()[1]
        contributions = []
        for idx in nonzero:
            word  = feature_names[idx]
            score = float(text_vec[0, idx] * coef[idx])
            if ' ' not in word and score > 0:          # single words, positive contribution
                contributions.append({'word': word, 'score': round(score, 4)})

        contributions.sort(key=lambda x: x['score'], reverse=True)
        return contributions[:n]
    except Exception:
        return []


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model():
    global pipeline
    if os.path.exists(MODEL_PATH):
        import joblib
        try:
            pipeline = patch_legacy_pipeline(joblib.load(MODEL_PATH))
            # Compatibility smoke test (sklearn version mismatch can fail at predict time)
            pipeline.predict_proba(["i feel happy"])
            print(f"[INFO] Model loaded from '{MODEL_PATH}'")
        except Exception as exc:
            pipeline = None
            print(f"[ERROR] Failed to use model '{MODEL_PATH}': {exc}")
            print("[WARN] Please run: python train.py")
    else:
        print(f"[WARN] Model not found at '{MODEL_PATH}'")
        print("[WARN] Please run:  python train.py")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({'error': 'Model not loaded. Run: python train.py'}), 503

    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error': 'Request body must contain a "text" field.'}), 400

    # Sanitise / limit input length
    raw_text = str(data['text'])[:1000].strip()
    if not raw_text:
        return jsonify({'error': 'Text field is empty.'}), 400

    text_clean   = preprocess_text(raw_text)
    probs        = pipeline.predict_proba([text_clean])[0]
    predicted_idx = int(np.argmax(probs))
    emotion       = EMOTION_LABELS[predicted_idx]
    cfg           = EMOTION_CONFIG[emotion]

    result = {
        'text':              raw_text,
        'predicted_emotion': emotion,
        'confidence':        round(float(probs[predicted_idx]), 4),
        'emoji':             cfg['emoji'],
        'color':             cfg['color'],
        'valence':           cfg['valence'],
        'arousal':           cfg['arousal'],
        'probabilities': {
            label: {
                'probability': round(float(p), 4),
                'emoji':       EMOTION_CONFIG[label]['emoji'],
                'color':       EMOTION_CONFIG[label]['color'],
                'valence':     EMOTION_CONFIG[label]['valence'],
                'arousal':     EMOTION_CONFIG[label]['arousal'],
            }
            for label, p in zip(EMOTION_LABELS, probs)
        },
        'top_words': get_top_words(text_clean, predicted_idx),
    }
    return jsonify(result)


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': pipeline is not None})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=int(os.environ.get('PORT', '5000')), host='127.0.0.1')
