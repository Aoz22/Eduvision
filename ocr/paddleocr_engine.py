import cv2
import numpy as np
from paddleocr import PaddleOCR

# we initialize the OCR engine once here so we don't reload the model every time we call run_ocr
ocr_engine = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    use_gpu=False,
    show_log=False,
)


def run_ocr(image_path):

    result = ocr_engine.ocr(image_path, cls=False)

    extracted = []

    if result and result[0]:
        for line in result[0]:
            bbox = line[0]   # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] | bounding box (top-left, top-right, bottom-right, bottom-left)
            text = line[1][0]
            confidence = round(line[1][1], 4)

            extracted.append({
                "text": text,
                "confidence": confidence,
                "bbox": bbox
            })

    return extracted


