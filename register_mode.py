import cv2
import os
from google.cloud import storage
from datetime import datetime
from PIL import Image, ImageTk

def run_register_mode(fakulte, bolum, sinif, isim, video_label):
    print("Kayıt başlatıldı. 's' ile fotoğraf çek, 'q' ile çık.")

    # Kamera başlat
    cap = cv2.VideoCapture(0)  # Gerekirse CAP_MSMF ya da CAP_DSHOW kullanılabilir
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    # GCS bağlantısı
    storage_client = storage.Client.from_service_account_json(
        r"C:\Users\enesb\Documents\GitHub\Real-Time-Face-Recognition-and-Attendance-System\real-time-attendance-460605-4367d4b382a9.json"
    )
    bucket = storage_client.bucket("dataset-aee")

    count = 1
    captured = False

    def capture_loop():
        nonlocal count, captured

        if not cap.isOpened():
            print("Kamera hala açık değil.")
            return

        ret, frame = cap.read()
        if not ret:
            print("Kare alınamadı.")
            video_label.after(10, capture_loop)
            return

        frame = cv2.flip(frame, 1)

        # GUI'de göster
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=im)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Fotoğraf çek
        if captured:
            filename = f"{isim}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            gcs_path = f"dataset/{fakulte}/{bolum}/{sinif}/{isim}/{count}.jpg"

            # Encode & upload
            _, buffer = cv2.imencode('.jpg', frame)
            blob = bucket.blob(gcs_path)
            blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')

            print(f"Fotoğraf yüklendi: {gcs_path}")
            count += 1
            captured = False  # Tek seferlik çekim

        video_label.after(10, capture_loop)

    def on_key_press(event):
        nonlocal captured
        if event.char == 's':
            print("Fotoğraf çekme tuşuna basıldı.")
            captured = True
        elif event.char == 'q':
            print("Kayıttan çıkılıyor.")
            cap.release()
            video_label.unbind("<Key>")
            return

    # Event bağla ve döngüyü başlat
    video_label.focus_set()
    video_label.bind("<Key>", on_key_press)
    capture_loop()
