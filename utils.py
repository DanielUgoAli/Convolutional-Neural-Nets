import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Some image processing functions

def load_image(image_path, size, show=False):
    original_img = Image.open(image_path)
    img = original_img.resize(size)
    img_array = np.array(img)
    # Normalize the image
    img_array = img_array / 255.0
    if show:
        img.show()
    return original_img, img_array

def random_crop(image, size, view=False):
    img_array = np.array(image)
    h, w, _ = img_array.shape
    new_h, new_w = size
    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)
    cropped_image = img_array[top: top + new_h, left: left + new_w]
    if view:
        # View the original image
        plt.figure()
        plt.imshow(cropped_image)
        plt.title("Cropped Image")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return cropped_image

def rotate_image(image, angle, view=False):
    img_array = np.array(image)
    # img_array = img_array / 255.0
    image_rotated = Image.fromarray(img_array).rotate(angle)
    if view:
        # View the original image
        plt.subplot(1, 2, 1)
        plt.imshow(img_array)
        plt.title("Original Image")
        plt.axis('off')
        # View the modified image
        plt.subplot(1, 2, 2)
        plt.imshow(image_rotated)
        plt.title("Rotated Image")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    return image_rotated



