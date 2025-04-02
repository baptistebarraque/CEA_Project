# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 08:46:55 2025

@author: natha
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 15:20:26 2025

@author: natha
"""
import os
import numpy as np
import re
import shutil

base_path = os.getcwd()  # Assure que le script fonctionne même si déplacé : on prend 
#celui sur lequel on travaille et on a les simulations
save_path = os.path.join(base_path, "donnees_simulation_2D")  # nom du dossier créé

# Création du dossier de destination s'il n'existe pas
os.makedirs(save_path, exist_ok=True)

# Parcours des simulations
for folder in os.listdir(base_path):
    if folder.startswith("CAS_SB_2D"):
        parts = folder.split('_')
        if len(parts) >= 6:  # le nom est correct en terme de quantité de caractères
            x = float(parts[-5])
            y = float(parts[-4])
            z = float(parts[-3])
            masse = float(parts[-2].replace('kg', ''))
            localisation = (x, y,z)
            pas = float(parts[-7]) # avoir le pas 
            # Accéder aux fichiers capteurs
            sensor_path = os.path.join(base_path, folder, "POST1D", "TE")
            if os.path.exists(sensor_path):
                for file in os.listdir(sensor_path):
                    if file.startswith("STATION_ST"):
                        capteur = int(file.replace("STATION_ST", ""))
                        nom_doc = os.path.join(sensor_path, file)
                        # Nouveau nom du fichier
                        new_name = f"{capteur}_{localisation[0]}_{localisation[1]}_{localisation[2]}_{masse}_{pas}.txt"
                        new_path = os.path.join(save_path, new_name)
                        
                        # Copier le fichier vers "donnees_simulation_2D"
                        shutil.copy(nom_doc, new_path)

print(f"✅ Tous les fichiers sont enregistrés dans {save_path}")
