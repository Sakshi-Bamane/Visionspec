import tensorflow as tf
import os
import cv2
from gradcam import apply_gradcam

model = tf.keras.models.load_model("models/mobilenet_qc.h5")

# pick one defect image
defect_dir = "data/raw/defect"
image_name = os.listdir(defect_dir)[0]
image_path = os.path.join(defect_dir, image_name)

output = apply_gradcam(image_path, model)

# save output instead of imshow
os.makedirs("outputs", exist_ok=True)
save_path = os.path.join("outputs", "gradcam_result.jpg")
cv2.imwrite(save_path, output)

print(f"✅ Grad-CAM image saved at: {save_path}")
