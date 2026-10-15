import cv2
from flask import Flask, render_template, request, redirect, url_for, session, Response
from ultralytics import YOLO
from flask import jsonify

# Inisialisasi Flask
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database user sederhana
USER_DB = {
    "admin": "1234",
    "user": "password"
}

# Muat model YOLOv8
model = YOLO('models/best.pt')

# ======================
# Detection & camera control
# ======================
detect_active = False
camera_active = True

# Fungsi untuk streaming video dari kamera
@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)  # Gunakan kamera default
        if not cap.isOpened():
            raise RuntimeError("Kamera tidak dapat diakses")

        while camera_active:  # Loop mengikuti flag camera_active
            ret, frame = cap.read()
            if not ret:
                break

            # Deteksi objek
            frame = detect_objects(frame)

            # Encode frame sebagai JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()  # Lepaskan kamera setelah loop selesai

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Definisikan detection box (area deteksi)
DETECTION_BOX = (150, 100, 500, 400)  # (x1, y1, x2, y2)

# Fungsi untuk mendeteksi objek dalam frame kamera
def detect_objects(frame):
    global detect_active

    # Jika detection STOP, kembalikan frame asli tanpa inferensi
    if not detect_active:
        return frame
    x1, y1, x2, y2 = DETECTION_BOX

    # Gambar area deteksi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Crop ROI
    roi = frame[y1:y2, x1:x2]

    # YOLO inference (confidence dinaikkan)
    results = model.predict(
        source=roi,
        imgsz=640,
        conf=0.65,
        verbose=False
    )

    if not results or results[0].boxes is None:
        return frame

    for box in results[0].boxes.data.tolist():
        x1_obj, y1_obj, x2_obj, y2_obj, conf, cls = box

        # Filter confidence rendah
        if conf < 0.65:
            continue

        # Hitung area bounding box (hindari noise)
        area = (x2_obj - x1_obj) * (y2_obj - y1_obj)
        if area < 2000:
            continue

        label = model.names[int(cls)]

        # Kembalikan koordinat ke frame asli
        x1_obj = int(x1_obj) + x1
        y1_obj = int(y1_obj) + y1
        x2_obj = int(x2_obj) + x1
        y2_obj = int(y2_obj) + y1

        # Gambar bounding box
        cv2.rectangle(
            frame,
            (x1_obj, y1_obj),
            (x2_obj, y2_obj),
            (0, 255, 0),
            2
        )

        # Tampilkan label
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1_obj, y1_obj - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return frame
# Halaman login
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in USER_DB and USER_DB[username] == password:
            session['user'] = username
            return redirect(url_for('run_page'))
        else:
            return render_template('login.html', error="Username atau password salah")
    return render_template('login.html')

# Halaman run
@app.route('/run')
def run_page():
    if 'user' in session:
        return render_template('run.html')
    return redirect(url_for('login'))

# Halaman index untuk deteksi
@app.route('/index')
def index_page():
    if 'user' in session:
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/start_detection')
def start_detection():
    global detect_active, camera_active
    detect_active = True
    camera_active = True
    return redirect(url_for('index_page'))

@app.route('/stop_detection')
def stop_detection():
    global detect_active
    detect_active = False
    return redirect(url_for('index_page'))

@app.route('/stop_camera')
def stop_camera():
    global camera_active
    camera_active = False
    return redirect(url_for('index_page'))

@app.route('/detection_status')
def detection_status():
    global detect_active
    return jsonify({"active": detect_active})


# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)