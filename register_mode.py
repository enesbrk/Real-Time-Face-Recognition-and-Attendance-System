import cv2
import os
from google.cloud import storage
from datetime import datetime
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, Button, Label

def run_register_mode(fakulte, bolum, sinif, isim, video_label=None):
    popup = tk.Toplevel()
    popup.title("Yüz Kaydı")
    popup.geometry("700x600")
    popup.resizable(False, False)

    lbl = Label(popup)
    lbl.pack(pady=10)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        messagebox.showerror("Hata", "Kamera açılamadı!")
        popup.destroy()
        return

    storage_client = storage.Client.from_service_account_json(
        r"C:\Users\enesb\Documents\GitHub\Real-Time-Face-Recognition-and-Attendance-System\real-time-attendance-460605-4367d4b382a9.json"
    )
    bucket = storage_client.bucket("dataset-aee")

    count = 1
    max_photos = 5

    current_frame = [None]  # Liste olarak tutulur, çünkü closure içinde değiştirilebilir

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            lbl.after(10, update_frame)
            return

        frame = cv2.flip(frame, 1)
        current_frame[0] = frame

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=im)
        lbl.imgtk = imgtk
        lbl.configure(image=imgtk)
        lbl.after(10, update_frame)

    def save_photo():
        nonlocal count
        if count > max_photos:
            return

        frame = current_frame[0]
        if frame is None:
            messagebox.showerror("Hata", "Kare alınamadı!")
            return

        gcs_path = f"dataset/{fakulte}/{bolum}/{sinif}/{isim}/{count}.jpg"
        _, buffer = cv2.imencode('.jpg', frame)
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')

        print(f"Fotoğraf {count} yüklendi -> {gcs_path}")
        count += 1

        if count > max_photos:
            finish_register()

    def finish_register():
        cap.release()
        popup.destroy()
        messagebox.showinfo("Kayıt", f"{isim} için yüz kaydı tamamlandı.")

    # Butonlar
    btn_frame = tk.Frame(popup)
    btn_frame.pack(pady=10)

    photo_btn = Button(btn_frame, text="📸 Fotoğraf Çek", command=save_photo, bg="#28a745", fg="white", font=("Arial", 12), width=20)
    photo_btn.pack(side="left", padx=10)

    quit_btn = Button(btn_frame, text="❌ İptal Et", command=finish_register, bg="#dc3545", fg="white", font=("Arial", 12), width=20)
    quit_btn.pack(side="right", padx=10)

    update_frame()
