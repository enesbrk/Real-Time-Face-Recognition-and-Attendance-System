import cv2
import os
from google.cloud import storage
from datetime import datetime
import numpy as np
import io
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import pickle

def run_register_mode():
    # GCS Ayarları
    GCS_CREDENTIALS = r"C:\\Users\\enesb\\Documents\\GitHub\\Real-Time-Face-Recognition-and-Attendance-System\\real-time-attendance-460605-4367d4b382a9.json"
    BUCKET_NAME = "dataset-aee"
    KNOWN_FACES_PATH = "known_faces.pkl"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=False, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # GCS Bağlantısı Kur
    storage_client = storage.Client.from_service_account_json(GCS_CREDENTIALS)
    bucket = storage_client.bucket(BUCKET_NAME)

    # Kullanıcıdan bilgi al
    fakulte = input("Fakülte adı (örn: Mühendislik ve Doğa Bilimleri Fakültesi): ").strip()
    bolum = input("Bölüm adı (örn: Yazılım Mühendisliği): ").strip()
    sinif = input("Sınıf (örn: 1. Sınıf): ").strip()
    isim = input("Ad Soyad (örn: Ali Yılmaz): ").strip()
    isim_kodu = isim.replace(" ", "_")
    gcs_key = f"{fakulte}/{bolum}/{sinif}/{isim_kodu}"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılmadı!")
        return

    print("Fotoğraf çekmek için hazırsın. 5 fotoğraf çekilecek...")
    foto_sayisi = 5
    cekilen = 0
    face_embeddings = []

    while cekilen < foto_sayisi:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Yüz Kaydı - 's' ile kaydet, 'q' ile çıkış", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            filename = f"{cekilen+1}.jpg"
            gcs_blob_path = f"dataset/{gcs_key}/{filename}"
            blob = bucket.blob(gcs_blob_path)

            success, encoded_image = cv2.imencode('.jpg', frame)
            if success:
                image_bytes = encoded_image.tobytes()
                blob.upload_from_string(image_bytes, content_type='image/jpeg')
                print(f"GCS'ye yüklendi: {gcs_blob_path}")

                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                face = mtcnn(img_pil)
                if face is not None:
                    embedding = model(face.unsqueeze(0).to(device)).detach().cpu().numpy()
                    face_embeddings.append(embedding)

                cekilen += 1
            else:
                print("Fotoğraf encode edilemedi.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if face_embeddings:
        mean_embedding = np.mean(face_embeddings, axis=0)
        if os.path.exists(KNOWN_FACES_PATH):
            with open(KNOWN_FACES_PATH, 'rb') as f:
                known_faces = pickle.load(f)
        else:
            known_faces = {}

        known_faces[gcs_key] = mean_embedding

        with open(KNOWN_FACES_PATH, 'wb') as f:
            pickle.dump(known_faces, f)

        print(f"{isim} kişisi known_faces.pkl dosyasına eklendi.")
    else:
        print("Yeterli yüz verisi elde edilemedi. known_faces.pkl güncellenmedi.")
