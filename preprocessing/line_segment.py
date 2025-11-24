import cv2
import numpy as np

def segment_lines(image_path, debug=False):
    """
    Adaptive line segmentation that handles various image types
    
    Args:
        image_path: Path to image file
        debug: If True, save debug images and print diagnostics
        
    Returns:
        List of line images (numpy arrays)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("[ERROR] Could not read image:", image_path)
        return []

    orig_h, orig_w = img.shape
    
    if debug:
        print(f"[DEBUG] Image dimensions: {orig_w}x{orig_h}")
        mean_val = np.mean(img)
        std_val = np.std(img)
        print(f"[DEBUG] Mean intensity: {mean_val:.2f}")
        print(f"[DEBUG] Std deviation: {std_val:.2f}")

    # Smooth noise - adaptive blur
    blur_kernel = max(3, min(7, orig_w // 200))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    img_blur = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    
    if debug:
        print(f"[DEBUG] Applied light blur (kernel={blur_kernel})")

    # Adaptive threshold - calculate block size based on image width
    block_size = max(11, min(101, orig_w // 20))
    if block_size % 2 == 0:
        block_size += 1
    
    # Use GAUSSIAN_C for better performance on printed text
    thresh = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Better for printed text
        cv2.THRESH_BINARY_INV,
        block_size,  # Adaptive
        10  # Adjusted C value
    )
    
    if debug:
        print(f"[DEBUG] Threshold saved (block_size={block_size})")
        cv2.imwrite("debug_threshold.png", thresh)

    # Vertical dilation to merge strokes into lines - adaptive kernel height
    kernel_height = max(15, min(30, orig_h // 40))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
    dilated = cv2.dilate(thresh, kernel, iterations=2)  # 2 iterations for better connectivity
    
    if debug:
        print(f"[DEBUG] Dilation saved (kernel_height={kernel_height})")
        cv2.imwrite("debug_dilated.png", dilated)

    # Find contours (lines)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        print(f"[DEBUG] Found {len(contours)} contours")

    line_images = []
    boxes = []

    # Adaptive minimum line height
    min_line_height = max(10, orig_h // 100)
    max_line_height = orig_h // 3  # Reject headers/titles that are too large
    min_line_width = orig_w * 0.05  # At least 5% of width

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # Adaptive filtering
        if min_line_height <= h <= max_line_height and w >= min_line_width:
            line_images.append(img[y:y+h, x:x+w])
            boxes.append((x, y, w, h))

    if debug:
        h_values = [h for _, _, _, h in boxes]
        w_values = [w for _, _, w, _ in boxes]
        if h_values:
            print(f"[DEBUG] Line height range: {min(h_values)}-{max(h_values)}")
            print(f"[DEBUG] Minimum line width: {min_line_width:.0f}")
        print(f"[DEBUG] Extracted {len(line_images)} valid lines")

    # Handle case where no lines detected - return full image as fallback
    if len(line_images) == 0:
        print(f"[WARNING] No text lines detected")
        return [img]

    # Sort top-to-bottom
    sorted_lines = [img for (_, img) in sorted(zip(boxes, line_images), key=lambda z: z[0][1])]

    return sorted_lines
