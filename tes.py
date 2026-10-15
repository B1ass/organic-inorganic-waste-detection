from ultralytics import YOLO

# Muat model YOLOv8 pre-trained
model = YOLO('yolov8n.pt')  # Gunakan YOLOv8 Nano untuk cepat

# Jalankan pelatihan
model.train(
    data='data.yaml',       # Path ke file konfigurasi YAML
    epochs=25,             # Jumlah epoch
    batch=16,               # Ukuran batch
    imgsz=640,              # Resolusi gambar
    patience=3,          # Berhenti otomatis jika tidak ada perbaikan
    flipud=0.0,             # Vertical flip augmentation (opsional)
    hsv_h=0.015,            # hue
    hsv_s=0.7,              # saturation
    hsv_v=0.4,              # value
    name='custom-yolov8'    # Nama eksperimen
)