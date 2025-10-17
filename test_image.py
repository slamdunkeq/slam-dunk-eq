from PIL import Image
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "shaq_dunk.png")
print(f"Script dir: {script_dir}")
print(f"Image path: {image_path}")
print(f"Image exists: {os.path.exists(image_path)}")

if os.path.exists(image_path):
    img = Image.open(image_path)
    print(f"Image size: {img.size}")
    print(f"Image mode: {img.mode}")
