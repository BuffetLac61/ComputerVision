import random
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np

class ImageGenerator:
    def __init__(self, height, width):
        """
        * height (int): heights of the generated images
        * width (int): widths of the generated images
        """
        self.height = height
        self.width = width

    def alter_perspective(self, image, angle):
        width, height = self.width, self.height
        image_np = np.array(image)
        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        shift = int(height * np.tan(np.radians(angle)))
        pts2 = np.float32([[shift, 0], [width - shift, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed = cv2.warpPerspective(image_np, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255, 0))
        return Image.fromarray(transformed)

    def add_noise_to_background(self, image, noise_level, block_size=10):
        np_image = np.array(image.convert('RGB'))  # Convert to RGB if necessary
        mask = np.all(np_image == [255, 255, 255], axis=-1)
        noise = np.random.randint(-noise_level, noise_level, (np_image.shape[0] // block_size, np_image.shape[1] // block_size, 3)).astype(np.int16)
        noise = np.repeat(np.repeat(noise, block_size, axis=0), block_size, axis=1)[:np_image.shape[0], :np_image.shape[1], :]
        np_image[mask] = np.clip(np_image[mask] + noise[mask], 0, 255).astype(np.uint8)
        return Image.fromarray(np_image)

    @staticmethod
    def ensure_in_box(image, box_size):
        image_np = np.array(image)
        non_empty_columns = np.where(image_np.max(axis=0) > 0)[0]
        non_empty_rows = np.where(image_np.max(axis=1) > 0)[0]
        if non_empty_columns.size and non_empty_rows.size:
            crop_box = (min(non_empty_columns), max(non_empty_columns), min(non_empty_rows), max(non_empty_rows))
            cropped_image = image.crop((crop_box[0], crop_box[2], crop_box[1] + 1, crop_box[3] + 1))
            if cropped_image.size[0] > box_size[0] or cropped_image.size[1] > box_size[1]:
                cropped_image.thumbnail(box_size, Image.Resampling.LANCZOS)
            result = Image.new("RGBA", box_size, (0, 0, 0, 0))
            result.paste(cropped_image, ((box_size[0] - cropped_image.size[0]) // 2, (box_size[1] - cropped_image.size[1]) // 2))
            return result
        return image

    def __call__(self, 
                char,
                char_x,
                char_y,
                char_color,
                font,  
                background_color,
                blur_level,
                rotation_angle,
                perspective_angle,
                noise_level):
        img_shape = (self.width, self.height)
        image = Image.new('RGBA', img_shape, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        char_y -= 35

        draw.text((char_x, char_y), char, fill=char_color, font=font)
        image = image.rotate(rotation_angle, expand=True, fillcolor=None)
        image = self.alter_perspective(image, perspective_angle)
    
        rotated_image = Image.new('RGBA', img_shape, background_color + (255,))
        rotated_image.paste(image, (0, 0), image)

        rotated_image = rotated_image.convert('RGB')
        rotated_image = self.ensure_in_box(rotated_image, img_shape)  # Ensure the image is within the box after transformations
        noisy_rotated_image = self.add_noise_to_background(rotated_image, noise_level)
        final_image = noisy_rotated_image.filter(ImageFilter.GaussianBlur(blur_level))
        return final_image
