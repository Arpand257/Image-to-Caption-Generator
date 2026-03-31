from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from utils import generate_caption, load_checkpoint, load_image


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pth"
UPLOAD_DIR = APP_DIR / "uploads"
DEVICE = "cpu"

UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)


def get_model_bundle():
    if not MODEL_PATH.exists():
        return None, None, None, f"Model file not found at {MODEL_PATH.name}. Run train.py first."

    try:
        model, vocab, config = load_checkpoint(str(MODEL_PATH), device=DEVICE)
        return model, vocab, config, None
    except Exception as exc:
        return None, None, None, f"Failed to load model: {exc}"


@app.route("/", methods=["GET", "POST"])
def home():
    caption = None
    error = None

    if request.method == "POST":
        model, vocab, _config, load_error = get_model_bundle()
        if load_error:
            error = load_error
        else:
            uploaded_file = request.files.get("file")
            if uploaded_file is None or uploaded_file.filename == "":
                error = "Please choose an image to caption."
            else:
                filename = secure_filename(uploaded_file.filename)
                saved_path = UPLOAD_DIR / filename
                uploaded_file.save(saved_path)
                image = load_image(saved_path, device=DEVICE)
                caption = generate_caption(model, image, vocab, device=DEVICE)

    return render_template("index.html", caption=caption, error=error)


if __name__ == "__main__":
    app.run(debug=True)
