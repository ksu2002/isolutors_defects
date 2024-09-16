import os
import shutil
import random

# Папка с исходными файлами
folderpath = 'generated_no_defect'

# Папки для сохранения файлов
train_images_dir = 'dataset/train/images'
train_labels_dir = 'dataset/train/labels'
valid_images_dir = 'dataset/valid/images'
valid_labels_dir = 'dataset/valid/labels'

# Создаем папки, если они не существуют
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(valid_images_dir, exist_ok=True)
os.makedirs(valid_labels_dir, exist_ok=True)

# Счетчик для определения, куда сохранять файлы
counter = 0

# Перебираем файлы в папке
for filename in os.listdir(folderpath):
    if filename.endswith('.jpg'):
        # Формируем пути для файлов
        src_file = os.path.join(folderpath, filename)
        dst_image_file = os.path.join(train_images_dir, filename)
        dst_label_file = os.path.join(train_labels_dir, filename.replace('.jpg', '.txt'))

        # Копируем файлы в папки
        shutil.copy(src_file, dst_image_file)
        with open(dst_label_file, 'w') as f:
            f.write('')

        # Каждую 10-ю пару файлов сохраняем в папку валидации
        counter += 1
        if counter % 10 == 0:
            # Перемещаем файлы в папку валидации
            shutil.move(dst_image_file, os.path.join(valid_images_dir, filename))
            shutil.move(dst_label_file, os.path.join(valid_labels_dir, filename.replace('.jpg', '.txt')))
