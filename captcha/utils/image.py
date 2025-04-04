import random
from PIL import Image, ImageDraw

def choose_random_position(h, w, char, font):
    image = Image.new('RGBA', (w,h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    char_bbox = draw.textbbox((0,0), char, font=font)
    char_width = char_bbox[2] - char_bbox[0]
    char_height = char_bbox[3] - char_bbox[1]
    base_position = ((w - char_width) // 2, (h - char_height) // 2)
    w_bound = (w - char_width) // 6
    offset_x = random.randint(-w_bound, w_bound) if w_bound > 0 else 0
    h_bound = (h - char_height) // 6
    offset_y = random.randint(-h_bound,h_bound) if h_bound > 0 else 0
    position = (base_position[0] + offset_x, base_position[1] + offset_y)
    return position

def concatenate_images(char_images):
    total_width = sum(i.size[0] for i in char_images)
    max_height = max(i.size[1] for i in char_images)
    concatenated_image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for image in char_images:
        concatenated_image.paste(image, (x_offset, 0))
        x_offset += image.size[0]
    return concatenated_image