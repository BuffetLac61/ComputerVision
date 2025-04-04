#!/home/alexis/dev/bin/python3
import torch
print(torch.cuda.is_available())  # Doit afficher True si tout est OK
print(torch.cuda.device_count())  # Nombre de GPUs détectés
print(torch.cuda.get_device_name(0))  # Nom de la carte utilisée