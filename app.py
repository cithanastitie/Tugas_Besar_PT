import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# =========================
# PATH AMAN (ANTI ERROR)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cyberbullying_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

model = None
vectorizer = None
load_error = None

# =========================
# LOAD MODEL & VECTORIZER
# =========================
def load_model_and_vectorizer():
    global load_error
    try:
        mdl = joblib.load(MODEL_PATH)
        vec = joblib.load(VECTORIZER_PATH)

        # VALIDASI TF-IDF SUDAH FIT
        if not hasattr(vec, "idf_"):
            raise ValueError(
                "TF-IDF vectorizer belum di-fit (idf_ tidak ditemukan). "
                "Pastikan vectorizer di-save setelah fit/fit_transform."
            )

        print("✅ Model dan TF-IDF vectorizer berhasil dimuat & valid!")
        return mdl, vec

    except Exception as e:
        load_error = str(e)
        print(f"❌ Error loading model/vectorizer: {e}")
        return None, None


model, vectorizer = load_model_and_vectorizer()

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html", error=load_error)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return render_template(
            "index.html",
            error=f"Model tidak siap digunakan: {load_error}"
        )

    try:
        text = request.form.get("text", "").strip()

        if not text:
            return render_template(
                "index.html",
                error="Masukkan teks terlebih dahulu!"
            )

        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]

        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(text_vectorized)[0]
            confidence = round(float(prediction_proba[prediction]) * 100, 2)
        else:
            confidence = 100.0

        label = "Bullying" if prediction == 1 else "Tidak Bullying"
        color = "danger" if prediction == 1 else "success"

        if prediction == 1:
            explanation = (
                "Teks mengandung indikasi bullying berdasarkan pola "
                "yang dipelajari oleh model machine learning."
            )
            suggestions = [
                "Hindari penggunaan kata-kata kasar atau menghina",
                "Gunakan bahasa yang lebih sopan dan konstruktif",
                "Pertimbangkan dampak kata-kata terhadap orang lain"
            ]
        else:
            explanation = (
                "Teks tidak mengandung indikasi bullying "
                "berdasarkan analisis model."
            )
            suggestions = [
                "Pertahankan komunikasi yang sehat",
                "Tetap perhatikan pilihan kata",
                "Hindari kata-kata yang berpotensi disalahartikan"
            ]

        return render_template(
            "result.html",
            text=text,
            prediction=label,
            confidence=confidence,
            color=color,
            explanation=explanation,
            suggestions=suggestions
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Terjadi error saat prediksi: {str(e)}"
        )


# =========================
# API ENDPOINT
# =========================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None or vectorizer is None:
        return jsonify({
            "error": "Model not ready",
            "detail": load_error
        }), 500

    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Text is required"}), 400

        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(text_vectorized)[0]
            confidence = float(probs[prediction])
        else:
            probs = [None, None]
            confidence = 1.0

        return jsonify({
            "text": text,
            "prediction": "Bullying" if prediction == 1 else "Not Bullying",
            "confidence": confidence,
            "probabilities": {
                "not_bullying": probs[0],
                "bullying": probs[1]
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# HEALTH CHECK
# =========================
@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy" if model and vectorizer else "unhealthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "vectorizer_fitted": hasattr(vectorizer, "idf_") if vectorizer else False,
        "error": load_error
    })


# =========================
# LOCAL RUN ONLY
# =========================
if __name__ == "__main__":
    app.run(debug=False)