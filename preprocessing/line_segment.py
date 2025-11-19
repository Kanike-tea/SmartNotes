import cv2
import numpy as np

def segment_lines(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 1. Denoise
    img_blur = cv2.GaussianBlur(img, (7, 7), 0)

    # 2. Otsu Threshold (invert)
    _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Connect text horizontally (increase width!)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 20))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # 4. Find contours → candidate lines
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Filter out noise
        if h < 25 or w < 100:
            continue

        line_img = img[y:y+h, x:x+w]
        lines.append(line_img)
        boxes.append((x, y, w, h))

    # 5. Sort by Y coordinate (top to bottom)
    sorted_lines = [line for _, line in sorted(zip(boxes, lines), key=lambda p: p[0][1])]

    return sorted_lines
