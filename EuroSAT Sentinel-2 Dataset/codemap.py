import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# === Конфигурация ===
dataset_path = "/Users/your/path/to/file"
image_size = (64, 64)  # Можно изменить

def load_dataset_from_filenames(path):
    X, y = [], []

    if not os.path.isdir(path):
        raise ValueError(f"Путь {path} не существует или не является директорией.")

    for filename in os.listdir(path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                # Предполагаем: метка — это всё до первого нижнего подчёркивания
                label = filename.split("_")[0]

                filepath = os.path.join(path, filename)
                image = Image.open(filepath).convert("RGB").resize(image_size)
                X.append(np.array(image).flatten())
                y.append(label)
            except Exception as e:
                print(f"❌ Ошибка с файлом {filename}: {e}")
    return np.array(X), np.array(y)

# === Загрузка данных ===
print("Загрузка данных...")
X, y = load_dataset_from_filenames(dataset_path)
print(f"Изображений: {len(X)}, Классов: {len(set(y))}")
print("Распределение классов:", Counter(y))

if len(X) == 0:
    raise ValueError("❌ Нет изображений для обучения.")

# === Разделение ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Обучение модели ===
print("Обучение модели...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Оценка ===
y_pred = model.predict(X_test)
print("Точность:", accuracy_score(y_test, y_pred))
print("Отчёт:\n", classification_report(y_test, y_pred))
