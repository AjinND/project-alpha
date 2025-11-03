import os
import urllib.request

os.makedirs("GroundingDINO/groundingdino/config", exist_ok=True)
os.makedirs("weights", exist_ok=True)

# Config
config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
if not os.path.exists(config_path):
    print("Downloading config...")
    urllib.request.urlretrieve(config_url, config_path)

# Weights
weights_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
weights_path = "weights/groundingdino_swint_ogc.pth"
if not os.path.exists(weights_path):
    print("Downloading weights (this may take a while)...")
    urllib.request.urlretrieve(weights_url, weights_path)

print("Grounding DINO setup complete!")