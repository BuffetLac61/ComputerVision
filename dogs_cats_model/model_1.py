#!/home/alexis/dev/bin/python3
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Vérifier CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
#device = "cpu"
input("Press Enter to continue...")

# Charger le feature extractor et le modèle
feature_extractor = ViTFeatureExtractor.from_pretrained('akahana/vit-base-cats-vs-dogs')
model = ViTForImageClassification.from_pretrained('akahana/vit-base-cats-vs-dogs')

# Envoyer le modèle sur le GPU si disponible
model.to(device)

# Mettre le modèle en mode évaluation
model.eval()


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ouvre l'image et s'assure qu'elle est RGB
    inputs = feature_extractor(images=image, return_tensors="pt")  # Prétraitement avec le feature extractor
    return inputs

def predict(image_path):
    inputs = preprocess_image(image_path)
    # Envoyer les inputs sur le même device que le modèle
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():  # Désactive le calcul des gradients (inutile pour l'inférence)
        outputs = model(**inputs)
    
    logits = outputs.logits  # Récupère les scores des classes
    predicted_class = torch.argmax(logits, dim=-1).item()  # Prend la classe avec le score le plus élevé
    
    # Le modèle est binaire (0 = chat, 1 = chien)
    labels = ["cat", "dog"]
    return labels[predicted_class]

def get_img_location_from_path (img_path : str) -> tuple:
# Utiliser une expression régulière pour extraire les coordonnées
    match = re.search(r'(\d+)_(\d+)\.jpg$', img_path)
    if match:
        x, y = int(match.group(1)), int(match.group(2))
        return (x, y)
    else:
        raise ValueError("Le chemin de l'image ne contient pas de coordonnées valides.")

def classify_and_locate_images_in_directory(directory_path):
    results = {}
    index = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Filtre les images
            image_localisation = get_img_location_from_path(filename)
            image_path = os.path.join(directory_path, filename)
            label = predict(image_path)
            results[image_localisation] = label
            print(f"{filename}: {label}")
            index += 1
            print(index)
            if index % 100000 == 0:
                image_exemple = Image.open(image_path)
                image_exemple.show()
                input("Press Enter to continue...")
    return results

directory_path = "/home/alexis/dev/ML_rootme/Flag"
results = classify_and_locate_images_in_directory(directory_path)


def create_and_display_image_from_dict(pixel_dict):
    # Déterminer la taille de l'image
    max_x = max(coord[0] for coord in pixel_dict.keys()) + 1
    max_y = max(coord[1] for coord in pixel_dict.keys()) + 1

    # Créer une image vide (noire)
    image = np.zeros((max_x, max_y))

    # Remplir l'image avec les pixels fournis
    for (x, y), label in pixel_dict.items():
        if label == "dog":
            image[x, y] = 1  # Blanc pour "dog"
        else:
            image[x, y] = 0  # Noir pour "cat"

    # Afficher l'image
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.show()

create_and_display_image_from_dict(results)