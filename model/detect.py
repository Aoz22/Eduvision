from ultralytics import YOLO
from pathlib import Path

def run_inference(image_path, model_path=r"C:\Projects\python\Eduvision\model\best.pt", confidence_threshold=0.25):
    image_path = Path(image_path)
    model_path = Path(model_path)
    
    model = YOLO(model_path)
    results = model.predict(image_path, conf=confidence_threshold)[0] # it handle one image at a time
    detection = []

    for box in results.boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        confidence = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detection.append({
            "class": class_name,
            "confidence": confidence,
            "bbox": (x1,y1,x2,y2)
        })

    return detection


