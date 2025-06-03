import cv2
import tkinter as tk
from tkinter import Button, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import os
from datetime import datetime
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from scipy.spatial.distance import cosine
from collections import defaultdict

# Model ve detector yükle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Embeddinglerin kaydedileceği klasör
EMBEDDINGS_DIR = "embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Bilinen yüzler sözlüğü: {isim: embedding numpy array}
def load_known_faces():
    known = {}
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            name = file[:-4]
            embedding = np.load(os.path.join(EMBEDDINGS_DIR, file))
            known[name] = embedding
    return known

known_faces = load_known_faces()
recorded_names = set()
verification_counts = defaultdict(int)
verification_threshold = 3

selected_mode = "attendance"

def set_register():
    global selected_mode
    selected_mode = "register"
    print("Register modu seçildi.")

def set_attendance():
    global selected_mode
    selected_mode = "attendance"
    print("Attendance modu seçildi.")

def compare_faces(embedding, known_faces):
    min_distance = float('inf')
    name = None
    for known_name, known_embedding in known_faces.items():
        dist = cosine(embedding.flatten(), known_embedding.flatten())
        if dist < min_distance:
            min_distance = dist
            name = known_name
    if min_distance < 0.6:  # eşik, kendine göre ayarlayabilirsin
        return name
    else:
        return "Tanımlanamayan Kişi"

def save_embedding(name, embedding):
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip()
    filename = os.path.join(EMBEDDINGS_DIR, safe_name + ".npy")
    np.save(filename, embedding)
    print(f"{name} kaydedildi.")

def add_attendance(name):
    if name in recorded_names:
        return
    recorded_names.add(name)
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Yoklama alındı: {name} - {current_time}")
    # Burada Firebase veya başka veritabanına ekleyebilirsin

def show_frame():
    global known_faces
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        root.after(10, show_frame)
        return

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    boxes, probs = mtcnn.detect(pil_img)

    draw = ImageDraw.Draw(pil_img)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = pil_img.crop((x1, y1, x2, y2)).resize((160, 160))
            face_tensor = transforms.ToTensor()(face).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(face_tensor).cpu().numpy()

            if selected_mode == "register":
                # Basit prompt: terminalden isim iste, GUI ile geliştirilebilir
                name = input("Kişi ismi: ").strip()
                if name:
                    save_embedding(name, embedding)
                    known_faces = load_known_faces()  # Güncelle
                    messagebox.showinfo("Kayıt", f"{name} başarıyla kayıt edildi.")
            else:
                name = compare_faces(embedding, known_faces)
                font = ImageFont.load_default()
                draw.rectangle(box, outline="red", width=3)
                draw.text((x1, y1 - 25), name, fill="red", font=font)

                if name != "Tanımlanamayan Kişi":
                    verification_counts[name] += 1
                    if verification_counts[name] >= verification_threshold:
                        add_attendance(name)
                        verification_counts[name] = 0
                else:
                    verification_counts.clear()

    imgtk = ImageTk.PhotoImage(image=pil_img)
    lbl.imgtk = imgtk
    lbl.configure(image=imgtk)
    lbl.after(10, show_frame)


root = tk.Tk()
root.title("FaceNet Attendance System")
root.geometry("1000x700")
root.configure(bg="#f0f4f7")

logo_label = tk.Label(root, text="🧠 AEEE", font=("Arial", 28, "bold"), fg="#005b96", bg="#f0f4f7")
logo_label.pack(anchor="nw", padx=20, pady=20)

btn_frame = tk.Frame(root, bg="#f0f4f7")
btn_frame.pack(side="left", anchor="n", padx=40, pady=100)

btn_register = Button(btn_frame, text="➕ Register", command=set_register,
                      bg="#007acc", fg="white", font=("Arial", 14, "bold"),
                      width=16, height=2, relief="flat", activebackground="#005b96")
btn_register.pack(pady=20, anchor="w")

btn_attendance = Button(btn_frame, text="✅ Attendance", command=set_attendance,
                        bg="#007acc", fg="white", font=("Arial", 14, "bold"),
                        width=16, height=2, relief="flat", activebackground="#005b96")
btn_attendance.pack(pady=20, anchor="w")

lbl = tk.Label(root, bg="#dbe9f4", bd=2, relief="ridge")
lbl.pack(expand=True, fill="both", padx=20, pady=20)

cap = cv2.VideoCapture(0)
show_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
