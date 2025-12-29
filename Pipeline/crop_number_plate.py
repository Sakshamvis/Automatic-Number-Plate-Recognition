from ultralytics import YOLO
import cv2
import os

# Load your trained ANPR YOLO model
model = YOLO("../RESULTS/anpr_yolo_gpu/weights/best.pt")

# Image folders (train + val)
image_folders = [
    "../DATA/final/images/train",
    "../DATA/final/images/val"
]

# Output folder for cropped plates
output_dir = "../RESULTS/cropped_plates"
os.makedirs(output_dir, exist_ok=True)

for folder in image_folders:
    split_name = os.path.basename(folder)  # train or val

    for img_name in os.listdir(folder):
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Skipping {img_name}")
                continue

            results = model(img, conf=0.25)

            for r in results:
                print(f"[{split_name}] {img_name} → detections: {len(r.boxes)}")

                for i, box in enumerate(r.boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    plate_crop = img[y1:y2, x1:x2]

                    save_path = os.path.join(
                        output_dir,
                        f"{split_name}_{os.path.splitext(img_name)[0]}_plate_{i}.jpg"
                    )

                    cv2.imwrite(save_path, plate_crop)

print(" Cropping finished for BOTH train and val")
