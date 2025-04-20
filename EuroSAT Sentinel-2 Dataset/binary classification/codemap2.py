import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Путь к папке с .jpg файлами
dataset_path = "/Users/your/file/path"

# Названия классов
classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

X = []
y = []

# Загружаем и размечаем изображения
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg"):
        filepath = os.path.join(dataset_path, filename)
        
        # Определим класс по имени файла
        label = None
        for cls in classes:
            if cls in filename:
                label = cls
                break
        if label is None:
            continue  # если класс не найден в названии, пропускаем
        
        # Загружаем и обрабатываем изображение
        try:
            img = Image.open(filepath).convert("RGB").resize((64, 64))
            img_array = np.array(img)
            if img_array.shape != (64, 64, 3):
                continue  # пропускаем битые изображения
            X.append(img_array.flatten())
            y.append(label)
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {e}")

# Преобразуем в numpy-массивы
X = np.array(X)
y = np.array(y)

# Кодируем метки в числа
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Делим на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Обучаем модель
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Оцениваем модель
y_pred = clf.predict(X_test)

print("Точность:", accuracy_score(y_test, y_pred))
print("Отчёт по классам:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
