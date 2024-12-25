import cv2
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import tkhtmlview

# Получаем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))

# Создаем новый распознаватель лиц
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Добавляем в него модель, которую мы обучили на прошлых этапах
recognizer.read(os.path.join(path, r'trainer', r'trainer.yml'))

# Указываем, что мы будем искать лица по примитивам Хаара
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + r"haarcascade_frontalface_default.xml")

# Получаем доступ к камере
cam = cv2.VideoCapture(0)
# Настраиваем шрифт для вывода подписей
font = cv2.FONT_HERSHEY_SIMPLEX

# Создаём словарь для сопоставления ID с именами
id_to_name = {1: 'Artem Borovkov', 2: 'Karina Zhubanazarova', 3: 'Diana Mishina',
              4: 'Sofya Nikolaeva', 5: 'Anton Hohlow', 6: 'Alexey Istomin',
              7: 'Alyona Rychkova', 8: 'Veronika Tenitskaya', 9: 'Artem Kuzmin',
              10: 'Nikita Lunegov', 11: 'Yulia Khristoforova', 12: 'Artem Borovkov',
              13: 'Alexey Istomin', 14: 'Anton Hohlow', 15: 'Artem Borovkov'}

# Функция для загрузки данных о человеке из файла
def load_person_data(name):
    file_path = os.path.join(path, r'GUI', f'{name}.html')
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return "<p>No data available</p>"

# Функция для обновления видео
def update_video():
    ret, frame = cam.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        nbr_predicted, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        if confidence < 100:
            name = id_to_name.get(nbr_predicted, 'Unknown')
            confidence_text = f"  {round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"  {round(100 - confidence)}%"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
        cv2.putText(frame, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, confidence_text, (x + 5, y + h + 25), font, 1, (255, 255, 0), 1)

        if confidence < 50:  # 100 - 50 = 50
            person_data = load_person_data(name)
            info_label.set_html(person_data)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_video)

# Функция для обработки нажатий клавиш
def on_key_press(event):
    if event.keysym in ['Return', 'Escape', 'space']:
        root.destroy()

# Создаем главное окно
root = tk.Tk()
root.title("Face Recognition")
root.attributes("-fullscreen", True)  # Делаем окно полноэкранным

# Устанавливаем значок окна
root.iconbitmap(r"C:\Users\User\Desktop\neuro\GUI\logo.jpg")  

# Устанавливаем цвет фона
root.configure(bg="#1f96f2")  

# Создаем фрейм для видео
video_frame = tk.Frame(root, bg="#1f96f2")  
video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=200, pady=400)

# Создаем виджет для отображения видео
video_label = tk.Label(video_frame, bg="#1f96f2")  # Вставьте сюда ваш цвет без альфа-канала
video_label.pack(fill=tk.BOTH, expand=True)

# Создаем фрейм для информации о человеке
info_frame = tk.Frame(root, bg="#1f96f2")  
info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=100, pady=300)

# Загружаем эмблему
logo_image = Image.open(r"C:\Users\User\Desktop\neuro\GUI\logo.jpg")  
logo_image = logo_image.resize((87, 79), Image.LANCZOS)  # Измените размер эмблемы по необходимости
logo_photo = ImageTk.PhotoImage(logo_image)

# Создаем виджет для отображения информации о человеке
info_label = tkhtmlview.HTMLLabel(info_frame, html="", width=500, height=600, bg="#1f96f2") 
info_label.pack(fill=tk.BOTH, expand=True)

# Создаем виджет для отображения эмблемы и размещаем его в верхней части фрейма
logo_label = tk.Label(info_frame, image=logo_photo, bg="#1f96f2")
logo_label.place(x=810, y=0, anchor="ne")  # Размещаем эмблему в левом верхнем углу

# Привязываем обработчик нажатий клавиш
root.bind("<KeyPress>", on_key_press)

# Запускаем обновление видео
update_video()

# Запускаем главный цикл tkinter
root.mainloop()

# Освобождаем ресурсы
cam.release()
cv2.destroyAllWindows()
