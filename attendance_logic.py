import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine
from datetime import datetime
from collections import defaultdict
import firebase_admin
from firebase_admin import credentials, db
from torchvision import transforms
from google.cloud import storage
from io import BytesIO
import json
import pickle

# --- Global variables ---
mtcnn = None
model = None
known_faces = {}
shown_messages = set()
verification_counts = defaultdict(int)
verification_threshold = 3

# Initialize all external connections and models
def init_attendance_logic():
    global mtcnn, model, known_faces

    # Firebase
    cred = credentials.Certificate(r"C:\Users\enesb\Documents\GitHub\Real-Time-Face-Recognition-and-Attendance-System\real-time-attendance-sys-13a15-firebase-adminsdk-fbsvc-0b0ea93420.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://real-time-attendance-sys-13a15-default-rtdb.europe-west1.firebasedatabase.app/'
        })

    # GCS
    storage_client = storage.Client.from_service_account_json(
        r"C:\Users\enesb\Documents\GitHub\Real-Time-Face-Recognition-and-Attendance-System\real-time-attendance-460605-4367d4b382a9.json"
    )
    bucket = storage_client.bucket("dataset-aee")

    mtcnn = MTCNN(keep_all=True)
    model = InceptionResnetV1(pretrained='vggface2').eval()

    # Load known faces
    if os.path.exists('known_faces.pkl'):
        with open('known_faces.pkl', 'rb') as f:
            known_faces = pickle.load(f)
        print("known_faces cache yüklendi.")
    else:
        known_faces = {}

    # Load and update from GCS
    CACHE_FILE = 'cached_faces.json'
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            processed_files = set(json.load(f))
    else:
        processed_files = set()

    new_processed_files = set()
    person_images = {}

    blobs = bucket.list_blobs(prefix="dataset/")
    for blob in blobs:
        if blob.name.endswith(".jpg") and blob.name not in processed_files:
            parts = blob.name.split("/")
            if len(parts) >= 5:
                fakulte, bolum, sinif, person_name = parts[1], parts[2], parts[3], parts[4]
                full_key = f"{fakulte}/{bolum}/{sinif}/{person_name}"

                img_bytes = blob.download_as_bytes()
                img = Image.open(BytesIO(img_bytes)).convert("RGB")

                if full_key not in person_images:
                    person_images[full_key] = []
                person_images[full_key].append(img)
                new_processed_files.add(blob.name)

    for person, images in person_images.items():
        embeddings = []
        for img in images:
            faces = mtcnn(img)
            if faces is not None:
                face_tensor = faces[0].unsqueeze(0)
                embedding = model(face_tensor).detach().numpy()
                embeddings.append(embedding)
        if embeddings:
            known_faces[person] = np.mean(embeddings, axis=0)

    with open(CACHE_FILE, 'w') as f:
        json.dump(list(processed_files.union(new_processed_files)), f)

    with open('known_faces.pkl', 'wb') as f:
        pickle.dump(known_faces, f)

    print("Veritabanı güncellendi.")

def sanitize(s):
    return s.replace('.', '_').replace('$', '_').replace('#', '_').replace('[', '_').replace(']', '_').replace('/', '_')

def add_attendance(full_key):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_key = now.strftime("%Y-%m-%d %H:%M:%S")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    parts = full_key.split("/")
    if len(parts) == 4:
        fakulte, bolum, sinif, isim = parts
        base_path = f"attendance/{sanitize(fakulte)}/{sanitize(bolum)}/{sanitize(sinif)}/{sanitize(isim)}"
        ref = db.reference(base_path)

        existing = ref.get()
        count_today = sum(1 for k in existing if k.startswith(date_str)) if existing else 0

        if count_today >= 10:
            if f"{isim}_limit" not in shown_messages:
                print(f"{isim} için günlük 10 kayıt sınırına ulaşıldı.")
                shown_messages.add(f"{isim}_limit")
            return

        ref.child(time_key).set({'status': 'Present', 'timestamp': timestamp})
        if f"{isim}_success" not in shown_messages:
            print(f"{isim} için yeni yoklama eklendi.")
            shown_messages.add(f"{isim}_success")

def compare_faces(embedding):
    min_distance = float('inf')
    name = "Tanımlanamayan Kişi"
    for known_name, known_embedding in known_faces.items():
        dist = cosine(embedding.flatten(), known_embedding.flatten())
        if dist < min_distance:
            min_distance = dist
            name = known_name
    return name if min_distance < 0.3 else "Tanımlanamayan Kişi"

def process_attendance_frame(frame):
    global verification_counts

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    boxes, _ = mtcnn.detect(pil_image)

    if boxes is not None:
        draw = ImageDraw.Draw(pil_image)
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(0, 255, 0), width=3)
            face = pil_image.crop((box[0], box[1], box[2], box[3])).resize((160, 160))

            face_tensor = transforms.ToTensor()(face).unsqueeze(0)
            embedding = model(face_tensor).detach().numpy()
            full_key = compare_faces(embedding)

            isim = full_key.split('/')[-1] if '/' in full_key else full_key
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            draw.text((box[0], box[1] - 25), isim, fill=(255, 0, 0), font=font)

            if full_key != "Tanımlanamayan Kişi":
                verification_counts[full_key] += 1
                if verification_counts[full_key] >= verification_threshold:
                    add_attendance(full_key)
                    verification_counts[full_key] = 0

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Exportable for GUI
__all__ = ["init_attendance_logic", "process_attendance_frame"]
