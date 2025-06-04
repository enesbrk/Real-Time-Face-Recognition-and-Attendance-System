import cv2
import tkinter as tk
from tkinter import Button, Entry, Label, Toplevel, messagebox
from PIL import Image, ImageTk
from attendance_logic import init_attendance_logic, process_attendance_frame
from register_mode import run_register_mode
import threading

selected_mode = "attendance"


def set_register():
    global selected_mode
    selected_mode = "register"
    print("Register modu seçildi.")
    open_register_popup()


def set_attendance():
    global selected_mode
    selected_mode = "attendance"
    print("Attendance modu seçildi.")
    start_attendance_mode()


def open_register_popup():
    popup = Toplevel()
    popup.title("Kayıt Bilgileri")

    tk.Label(popup, text="Fakülte:").grid(row=0, column=0, padx=10, pady=5)
    fakulte_entry = Entry(popup)
    fakulte_entry.grid(row=0, column=1, padx=10, pady=5)

    tk.Label(popup, text="Bölüm:").grid(row=1, column=0, padx=10, pady=5)
    bolum_entry = Entry(popup)
    bolum_entry.grid(row=1, column=1, padx=10, pady=5)

    tk.Label(popup, text="Sınıf:").grid(row=2, column=0, padx=10, pady=5)
    sinif_entry = Entry(popup)
    sinif_entry.grid(row=2, column=1, padx=10, pady=5)

    tk.Label(popup, text="Ad Soyad:").grid(row=3, column=0, padx=10, pady=5)
    isim_entry = Entry(popup)
    isim_entry.grid(row=3, column=1, padx=10, pady=5)

    def submit():
        fakulte = fakulte_entry.get()
        bolum = bolum_entry.get()
        sinif = sinif_entry.get()
        isim = isim_entry.get()

        if not all([fakulte, bolum, sinif, isim]):
            messagebox.showerror("Hata", "Lütfen tüm alanları doldurun.")
            return

        popup.destroy()
        run_register_mode(fakulte, bolum, sinif, isim)

    tk.Button(popup, text="Kaydet", command=submit, bg="#007acc", fg="white", font=("Arial", 12)).grid(row=4, column=0, columnspan=2, pady=10)


attendance_cap = None
attendance_running = False
attendance_thread = None

def start_attendance_mode():
    global attendance_cap, attendance_running, attendance_thread
    if attendance_running:
        return
    init_attendance_logic()
    attendance_cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    if not attendance_cap.isOpened():
        messagebox.showerror("Hata", "Kamera açılamadı!")
        return
    attendance_running = True
    attendance_thread = threading.Thread(target=process_attendance_frame_stream)
    attendance_thread.start()

def stop_attendance_mode():
    global attendance_cap, attendance_running
    attendance_running = False
    if attendance_cap:
        attendance_cap.release()
        attendance_cap = None
    
    # Ekranı sıfırla
    lbl.configure(image='')  # Görseli kaldır
    lbl.imgtk = None         # Bellekteki görüntü referansını sil  


def process_attendance_frame_stream():
    global attendance_cap, attendance_running
    while attendance_running:
        ret, frame = attendance_cap.read()
        if not ret or frame is None:
            continue
        frame = cv2.flip(frame, 1)
        processed = process_attendance_frame(frame)
        img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=im)
        lbl.imgtk = imgtk
        lbl.configure(image=imgtk)
        lbl.update()


root = tk.Tk()
root.title("AEEE - Attendance and Registration System")
root.geometry("1000x700")
root.configure(bg="#f0f4f7")

logo_label = Label(root, text="🧠 AEEE", font=("Arial", 28, "bold"), fg="#005b96", bg="#f0f4f7")
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

btn_stop = Button(btn_frame, text="🛑 Kamerayı Kapat", command=stop_attendance_mode,
                  bg="#cc0000", fg="white", font=("Arial", 14, "bold"),
                  width=16, height=2, relief="flat", activebackground="#990000")
btn_stop.pack(pady=20, anchor="w")


lbl = tk.Label(root, bg="#dbe9f4", bd=2, relief="ridge")
lbl.pack(expand=True, fill="both", padx=20, pady=20)

root.mainloop()

if attendance_cap:
    attendance_cap.release()
cv2.destroyAllWindows()
