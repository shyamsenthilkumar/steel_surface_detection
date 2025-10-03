import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# -----------------------------
# 1. Load trained model
# -----------------------------
model = load_model("steel_defect_cnn.h5", compile=False)

# Define your defect class names
class_names = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]

# Get model input shape
input_shape = model.input_shape
HEIGHT, WIDTH, CHANNELS = input_shape[1], input_shape[2], input_shape[3]

# -----------------------------
# 2. Flask configuration
# -----------------------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------------
# 3. Helper functions
# -----------------------------
def prepare_image(filepath):
    """Preprocess uploaded image"""
    img = image.load_img(filepath, target_size=(HEIGHT, WIDTH))
    img_array = image.img_to_array(img)

    if CHANNELS == 1:
        img_array = cv2.cvtColor(img_array.astype("uint8"), cv2.COLOR_RGB2GRAY)
        img_array = np.expand_dims(img_array, axis=-1)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_defect(filepath):
    """Run prediction on image file"""
    img_array = prepare_image(filepath)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_names[predicted_class]


# -----------------------------
# 4. Flask Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            label = predict_defect(filepath)
            return render_template("result.html", image_file=filepath, label=label)

    return render_template("upload.html")


# -----------------------------
# 5. Live Camera Streaming
# -----------------------------
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ No camera detected.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Preprocess frame
        img = cv2.resize(frame, (WIDTH, HEIGHT))
        if CHANNELS == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)
        img_array = img / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        label = class_names[predicted_class]

        # Overlay label on frame
        cv2.putText(frame, f"Defect: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route("/camera")
def camera():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# -----------------------------
# 6. Health Check (for Render)
# -----------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# -----------------------------
# 7. Run the App (Gunicorn-ready)
# -----------------------------
if __name__ == "__main__":
    # For local testing only
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
