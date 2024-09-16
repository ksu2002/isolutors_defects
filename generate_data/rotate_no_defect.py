'''
Скрипт для создания повернутых деталей(без дефектов).
'''
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from generate_data.rotate_utils import rotate_image, crop_around_center, largest_rotated_rect


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def main():
    isulator= cv2.imread('perfect/round_no_metal.jpg')
    template = cv2.cvtColor(isulator, cv2.COLOR_BGR2RGB)
    W, H= template.shape[:2]
    count = 1
    print('start')
    for angle in frange(0.0, 360.0, 0.5):
        try:
            print(count)
            rotated_template = rotate_image(template, angle)
            image_rotated_cropped = crop_around_center(
                    rotated_template,
                    *largest_rotated_rect(
                        W,
                        H,
                        math.radians(angle)
                    ))

            w, h = image_rotated_cropped.shape[:2]
            center_x = w // 2
            center_y = h // 2

            # Вычисление размеров обрезанного изображения
            crop_size = 425

            # Обрезка изображения
            cropped_image = image_rotated_cropped[center_y - crop_size:center_y + crop_size,
                                center_x - crop_size:center_x + crop_size]
            w, h = cropped_image.shape[:2]
            if w<800:
                    continue
            if h<800:
                    continue
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)    
            filename = f'generated_no_defect/perf_round_no_metal_{count}.jpg'
            count += 1
            cv2.imwrite(filename, cropped_image)     
        except Exception:
            continue
        
if __name__ == "__main__":
    main()        