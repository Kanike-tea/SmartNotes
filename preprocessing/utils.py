# utils.py — (Anushka Backend)

import os

def is_image(file_path):
    return file_path.lower().endswith((".jpg", ".jpeg", ".png"))

def list_images(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if is_image(f)]

