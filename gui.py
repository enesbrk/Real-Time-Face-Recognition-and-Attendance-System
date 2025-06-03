import cv2
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageTk

selected_mode = "attendance"

def set_register():
    global selected_mode
    selected_mode = "register"
    print("Register modu seçildi.")

def set_attendance():
    global selected_mode
    selected_mode = "attendance"
    print("Attendance modu seçildi.")

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lbl.imgtk = imgtk
    lbl.configure(image=imgtk)
    lbl.after(10, show_frame)

# Ana pencere
root = tk.Tk()
root.title("FaceNet Attendance System")
root.geometry("1000x700")
root.configure(bg="#f0f4f7")  # Açık gri arka plan

# Sol üst köşede logo / başlık
logo_label = tk.Label(root, text="🧠 AEEE", font=("Arial", 28, "bold"), fg="#005b96", bg="#f0f4f7")
logo_label.pack(anchor="nw", padx=20, pady=20)

# Sol panel (butonlar)
btn_frame = tk.Frame(root, bg="#f0f4f7")
btn_frame.pack(side="left", anchor="n", padx=40, pady=100)

# Register butonu
btn_register = Button(btn_frame, text="➕ Register", command=set_register,
                      bg="#007acc", fg="white", font=("Arial", 14, "bold"),
                      width=16, height=2, relief="flat", activebackground="#005b96")
btn_register.pack(pady=20, anchor="w")

# Attendance butonu
btn_attendance = Button(btn_frame, text="✅ Attendance", command=set_attendance,
                        bg="#007acc", fg="white", font=("Arial", 14, "bold"),
                        width=16, height=2, relief="flat", activebackground="#005b96")
btn_attendance.pack(pady=20, anchor="w")

# Kamera görüntüsü
lbl = tk.Label(root, bg="#dbe9f4", bd=2, relief="ridge")
lbl.pack(expand=True, fill="both", padx=20, pady=20)

# Kamera başlat
cap = cv2.VideoCapture(1)
show_frame()

# GUI döngüsü
root.mainloop()

# Kamera kapat
cap.release()
cv2.destroyAllWindows()
