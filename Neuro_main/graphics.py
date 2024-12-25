import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Получаем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))

# Создаем новый распознаватель лиц
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Добавляем в него модель, которую мы обучили на прошлых этапах
recognizer.read(path + '/trainer/trainer.yml')

# Указываем, что мы будем искать лица по примитивам Хаара
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Загружаем тестовый набор данных
def load_test_data(test_path):
    test_images = []
    test_labels = []
    for root, dirs, files in os.walk(test_path):
        for file in files:
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                label = int(os.path.split(root)[1])
                test_images.append(img)
                test_labels.append(label)
    return test_images, test_labels

# Тестируем модель на тестовом наборе данных
def test_model(test_images, test_labels):
    predicted_labels = []
    for img in test_images:
        label, confidence = recognizer.predict(img)
        predicted_labels.append(label)
    return predicted_labels

# Путь к тестовому набору данных
test_path = path + '/test'

# Загружаем тестовый набор данных
test_images, test_labels = load_test_data(test_path)

# Тестируем модель
predicted_labels = test_model(test_images, test_labels)

# Вычисляем точность
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Строим матрицу ошибок
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Визуализируем матрицу ошибок
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Визуализируем точность
plt.figure(figsize=(10, 7))
plt.bar(["Accuracy"], [accuracy * 100])
plt.ylim(0, 100)
plt.ylabel("Percentage")
plt.title("Model Accuracy")
plt.show()
