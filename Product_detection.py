import os
import cv2
from ultralytics import YOLO

# Paths
model_path = r"D:\Ravi_Kumar_hans\self\Demo\soap_best.pt"
input_folder = r"D:\Ravi_Kumar_hans\self\Demo\images"
output_root = r"D:\Ravi_Kumar_hans\self\Product-Detection-using-YOLOv8\output_1"

# Load model
model = YOLO(model_path)
class_names = model.names

# Create output root
os.makedirs(output_root, exist_ok=True)

# Loop through images
for file_name in os.listdir(input_folder):
    if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(input_folder, file_name)
    img = cv2.imread(image_path)

    results = model(img)

    for result in results:
        for idx, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = class_names[cls_id]

            # Create class folder
            class_folder = os.path.join(output_root, class_name)
            os.makedirs(class_folder, exist_ok=True)

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop detection
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Save cropped detection
            base_name = os.path.splitext(file_name)[0]
            save_name = f"{base_name}_{idx}.jpg"
            save_path = os.path.join(class_folder, save_name)

            cv2.imwrite(save_path, crop)

    print(f"Processed: {file_name}")

print("✅ Only detections saved (cropped) by class name!")
