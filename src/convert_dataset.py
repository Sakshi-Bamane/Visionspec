import os
import shutil

SOURCE = "pcb-defect-dataset/train"
TARGET = "data/raw"

images_path = os.path.join(SOURCE, "images")
labels_path = os.path.join(SOURCE, "labels")

os.makedirs(os.path.join(TARGET, "pass"), exist_ok=True)
os.makedirs(os.path.join(TARGET, "defect"), exist_ok=True)

for img_name in os.listdir(images_path):
    img_path = os.path.join(images_path, img_name)
    label_path = os.path.join(labels_path, img_name.replace(".jpg", ".txt"))

    # DEFECT if label file exists and is not empty
    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        shutil.copy(img_path, os.path.join(TARGET, "defect", img_name))
    else:
        shutil.copy(img_path, os.path.join(TARGET, "pass", img_name))

print("✅ Dataset successfully converted to PASS / DEFECT format")
