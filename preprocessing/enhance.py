import cv2
import numpy as np

def deskew(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 45:
            angles.append(angle)

    if len(angles) == 0:
        return image

    average_angle = np.mean(angles)

    if abs(average_angle) < 0.5:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, average_angle, 1.0)
    corrected = cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))

    return corrected

## this for enhance contrasts (the darkest arae)
def apply_clahe(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced

def denoise(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    return denoised

def remove_background(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)
    
    padding = 10
    x = max(0, x + padding)
    y = max(0, y + padding)
    w = min(image.shape[1], x + w - padding) - x
    h = min(image.shape[0], y + h - padding) - y

    cropped = image[y:y+h, x:x+w]

    return cropped