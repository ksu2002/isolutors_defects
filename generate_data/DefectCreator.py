import os
import random
from random import choice, randint, uniform

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from PIL import Image, ImageEnhance


class DefectCreator:
    """
    Параметры класса
    png_images_dir - папка с png картинками, название файлов должно начинаться на название класса
    class_name - название класса
    darkens_factor - коэффициент затемнения картинки.
    brgh_factor - коэффициень яркости картинки
    min_blur_intensity - минимальное размытие картинки
    max_blur_intensity - максимальное размытие картинки
    transparency - прозрачность
    """

    def __init__(
        self,
        png_images_dir: str,
        class_name: str,
        darkens_factor: float = 1,
        brgh_factor: float = 1,
        min_blur_intensity: int = 1,
        max_blur_intensity: int = 1,
        blur_png_intensity: int = 1,
        range_random_degree: tuple | None = (0, 100),
        min_count_to_paste: int = 1,
        max_count_to_paste: int = 1,
        ration_range: tuple = (50, 100),
        transparency: tuple = (1.0, 1.0),
        mask: ndarray | None = np.zeros(5),
    ) -> None:
        # путь к папке с подпапками соответствующие названию классов
        self.png_images_dir = png_images_dir
        # путь к папке с png картинками
        self.png_files_list = [
            file
            for file in os.listdir(self.png_images_dir)
            if file.startswith(class_name)
        ]
        self.range_random_degree = range_random_degree
        # максимальное количество накладываемых объектов
        self.min_count_to_paste = min_count_to_paste
        self.max_count_to_paste = max_count_to_paste
        # настройки яркости и размытия
        self.darkens_factor = darkens_factor
        self.brgh_factor = brgh_factor
        self.min_blur_intensity = min_blur_intensity
        self.max_blur_intensity = max_blur_intensity
        # размытие накладываемого изображения
        self.blur_png_intensity = blur_png_intensity
        # масштаб изменения размера вставляемого шаблона
        # self.min_ration = 60
        # self.max_ration = 80
        self.ration_range = ration_range
        self.transparency = transparency
        self.class_name = class_name

    # наложение png шаблона на фон
    def __call__(self, back_image: Image, mask) -> Image:
        self.mask = mask
        new_image, x1, y1, x2, y2 = self._paste_objects_on_back(back_image=back_image)
        return new_image, x1, y1, x2, y2

    def _change_transparency(self, image, transparency):
        r, g, b, a = image.split()
        a = ImageEnhance.Brightness(a).enhance(
            transparency
        )  # Изменяем яркость альфа-канала
        return Image.merge("RGBA", (r, g, b, a))

    def _paste_objects_on_back(self, back_image: Image) -> Image:
        rnd_cnt = randint(self.min_count_to_paste, self.max_count_to_paste)
        for i in range(rnd_cnt):
            rnd_sample = choice(self.png_files_list)
            png_image = Image.open(os.path.join(self.png_images_dir, rnd_sample))
            if self.range_random_degree:
                rand_rotate_degree = randint(
                    self.range_random_degree[0], self.range_random_degree[1]
                )
                png_image = png_image.rotate(rand_rotate_degree, expand=True)

            max_size = self._calculate_max_image_size(
                back_image_size=back_image.size, top_image_size=png_image.size
            )
            rand_size = self._calculate_random_size(max_size)
            rand_x, rand_y = self._calculate_random_coordinates(
                rand_size, back_image.size
            )
            png_image = png_image.resize(rand_size)
            png_image = png_image.convert("RGBA")
            back_image.paste(png_image, (rand_x, rand_y), mask=png_image)

            # mask_image = Image.fromarray(self.mask*255)
            # mask_image = mask_image.convert("RGBA")
            # back_image.paste(mask_image, (0, 0), mask=mask_image)

            x1 = rand_x - 3
            y1 = rand_y - 3
            x2 = rand_x + 3 + rand_size[0]
            y2 = rand_y + 3 + rand_size[1]

        return back_image, x1, y1, x2, y2

    def _calculate_random_size(self, max_size: tuple) -> tuple:
        rand_ratio = randint(*self.ration_range) / 100
        rand_size = [int(max_size[0] * rand_ratio), int(max_size[1] * rand_ratio)]

        return rand_size

    def _calculate_random_coordinates(self, rand_size: tuple, back_image_size) -> tuple:
        if self.class_name == "fon" or self.class_name == "metall" or self.class_name == "crack":
            boundaries = np.argwhere(np.abs(np.diff(self.mask)) == 1)
            x, y = boundaries[:, 1], boundaries[:, 0]
            random_x, random_y = random.choice(x), random.choice(y)
            
            if random_x - rand_size[0] / 2 < 0:
                print(random_x)
                print( rand_size[0] / 2)
                x = int(random_x)
            else:
                x = int(random_x - rand_size[0] / 2)

            if random_y - rand_size[1] / 2 < 0:
                y = int(random_y)
            else:
                y = int(random_y - rand_size[1] / 2)     
                
            # print(random_y, random_x)
            # x = int(random_x - rand_size[0]/2)
            # y = int(random_y - rand_size[1]/2)

            # x = random_x
            # y = random_y
            
            return x, y

        indices = np.argwhere(self.mask == 1)
        random_index = np.random.choice(indices.shape[0], size=1, replace=False)
        x, y = indices[random_index][0]
        while not (np.all(self.mask[x : x + rand_size[0], y : y + rand_size[1]] == 1)):
            random_index = np.random.choice(indices.shape[0], size=1, replace=False)
            x, y = indices[random_index][0]
        return y, x

    def _calculate_max_image_size(
        self,
        back_image_size: tuple,
        top_image_size: tuple,
    ) -> tuple:
        width, height = top_image_size  # максимальный размер
        back_ratio = width / height
        img_ratio = top_image_size[0] / top_image_size[1]

        if back_ratio >= 1:  # альбомная ориентация фона
            if img_ratio <= back_ratio:  # вставляемая картинка уже чем фон
                width_new = int(height * img_ratio)
                size_new = width_new, height
            else:  # вставляемая картинка шире чем фон
                height_new = int(width / img_ratio)
                size_new = width, height_new
        else:  # книжная ориентация фона
            if img_ratio >= back_ratio:  # вставляемая картинка ниже чем фон
                height_new = int(width / img_ratio)
                size_new = width, height_new
            else:  # вставляемая картинка выше чем фон
                width_new = int(height * img_ratio)
                size_new = width_new, height

        return size_new

    def _change_brightness(self, img: Image) -> Image:
        enhancer = ImageEnhance.Brightness(img)
        factor = uniform(self.darkens_factor, self.brgh_factor)
        im_output = enhancer.enhance(factor)
        return im_output


overlayer = DefectCreator(
    png_images_dir="./templates",
    class_name="metall",
    min_count_to_paste=1,
    max_count_to_paste=1,
    range_random_degree=(0, 90),
    brgh_factor=3.0,
    darkens_factor=1.0,
    blur_png_intensity=0.0,
)


def add_defects(image: Image, obj_mask_resized) -> Image:
    image, x1, y1, x2, y2 = overlayer(image, obj_mask_resized)
    return image, x1, y1, x2, y2
