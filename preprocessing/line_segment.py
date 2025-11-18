import cv2
import numpy as np

def segment_lines(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("[ERROR] Could not read image:", image_path)
        return []

    # Smooth noise
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive threshold for handwritten notes
    thresh = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25, 15
    )

    # Vertical dilation to merge strokes into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours (lines)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_images = []
    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 20:                 # ignore noise
            line_images.append(img[y:y+h, x:x+w])
            boxes.append((x, y, w, h))

    # Sort top-to-bottom
    sorted_lines = [img for (_, img) in sorted(zip(boxes, line_images), key=lambda z: z[0][1])]

    return sorted_lines
