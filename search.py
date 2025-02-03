from pathlib import Path
import clip
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)

def get_file_paths(directory):
    return [str(file) for file in Path(directory).iterdir() if file.is_file()]

# Example usage
directory_path = "photos"
file_paths = get_file_paths(directory_path)

# Load and preprocess images
images = [preprocess(Image.open(f)).unsqueeze(0).to(device) for f in file_paths]
images = torch.cat(images, dim=0)  # Stack images into a single tensor

# Get user input for text query
user_input = input("Enter the description of the image you are looking for: ")
text = clip.tokenize([user_input]).to(device)

# Encode images and text
with torch.no_grad():
    image_features = model.encode_image(images)
    text_features = model.encode_text(text)

# Calculate similarity
similarities = (image_features @ text_features.T).squeeze()
best_match_idx = similarities.argmax().item()

print(f"Best matching image is: {file_paths[best_match_idx]}")
