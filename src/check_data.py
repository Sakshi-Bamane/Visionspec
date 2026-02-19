import os

base = "data/raw"

for cls in os.listdir(base):
    path = os.path.join(base, cls)
    print(f"{cls}: {len(os.listdir(path))} images")
