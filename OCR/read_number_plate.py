import easyocr
import cv2
import os
import csv

# Initialize EasyOCR (English only)
reader = easyocr.Reader(['en'], gpu=False)

# Input: cropped number plate images
input_dir = "../RESULTS/cropped_plates"

# Output CSV
output_csv = "../RESULTS/anpr_results.csv"

# Simple preprocessing for blurry plates
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

results = []

for img_name in os.listdir(input_dir):
    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        processed = preprocess(img)

        ocr_result = reader.readtext(processed, detail=0)

        plate_text = " ".join(ocr_result).replace(" ", "").upper()

        results.append([img_name, plate_text])

        print(f"{img_name} → {plate_text}")

# Save results to CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "plate_number"])
    writer.writerows(results)

print("✅ OCR complete. Results saved to anpr_results.csv")
