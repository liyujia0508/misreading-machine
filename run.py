"""
run.py – One-click launcher
  1. Checks that the model file exists (if not, trains it first)
  2. Starts the Flask web application on http://localhost:5000
"""

import os
import subprocess
import sys
import webbrowser
import threading
import time
import socket
import warnings


MODEL_PATH = 'model/emotion_classifier.pkl'


def patch_legacy_pipeline(model_pipeline):
    """Patch known sklearn compatibility fields for older/newer pickles."""
    try:
        clf = model_pipeline.named_steps.get('clf')
        if clf is not None and not hasattr(clf, 'multi_class'):
            clf.multi_class = 'auto'
    except Exception:
        pass
    return model_pipeline


def find_available_port(start_port: int = 5000, max_tries: int = 20) -> int:
    """Find an available local TCP port, starting from start_port."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return port
    raise RuntimeError(f"No available port in range {start_port}-{start_port + max_tries - 1}")


def train_if_needed():
    if os.path.exists(MODEL_PATH):
        if is_model_usable(MODEL_PATH):
            print(f"[✓] Model found at '{MODEL_PATH}'")
            return
        print("[!] Existing model is incompatible with current environment.")
        print("[!] Re-training model to fix compatibility …\n")
    else:
        print("[!] Model not found – starting training …\n")

    result = subprocess.run(
        [sys.executable, 'train.py'],
        check=False,
    )
    if result.returncode != 0:
        print("[✗] Training failed. Check the output above.")
        sys.exit(1)
    print()


def is_model_usable(model_path: str) -> bool:
    """Return True when the serialized model can run predict_proba safely."""
    try:
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipeline = patch_legacy_pipeline(joblib.load(model_path))
        pipeline.predict_proba(["i feel happy"])
        return True
    except Exception:
        return False


def open_browser(port: int):
    time.sleep(1.5)
    # Safari can be picky with localhost resolution/proxy settings on some Macs.
    webbrowser.open(f'http://127.0.0.1:{port}')


def main():
    print()
    print("=" * 54)
    print("  Emotion Recognition Demo  ·  Affective Computing")
    print("=" * 54)

    train_if_needed()

    preferred_port = int(os.environ.get('PORT', '5000'))
    port = find_available_port(preferred_port)
    if port != preferred_port:
        print(f"[i] Port {preferred_port} is occupied, switched to {port}")

    print(f"[→] Starting Flask server on http://127.0.0.1:{port}")
    print("[→] Press Ctrl+C to quit\n")

    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    os.environ.setdefault('FLASK_ENV', 'production')
    from app import app, load_model
    load_model()
    app.run(debug=False, port=port, host='127.0.0.1')


if __name__ == '__main__':
    main()
