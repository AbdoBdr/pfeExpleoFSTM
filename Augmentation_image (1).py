import os
import cv2
import numpy as np

# Chemin du répertoire contenant les dossiers de données originales
input_dir = r"C:\Users\Dell\Downloads\testData"
# Chemin du répertoire où les images augmentées seront sauvegardées
output_dir = r"C:\Users\Dell\Downloads\testData_Augmented"

# Créer le répertoire de sortie s'il n'existe pas déjà
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parcourir tous les dossiers dans le répertoire d'entrée
for folder_name in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder_name)
    if os.path.isdir(folder_path):  # Assurez-vous qu'il s'agit d'un dossier
        # Créer un sous-répertoire dans le répertoire de sortie pour chaque dossier de données
        output_subdir = os.path.join(output_dir, folder_name)
        os.makedirs(output_subdir, exist_ok=True)

        # Parcourir toutes les images dans le dossier de données
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Assurez-vous que vous ne travaillez qu'avec des images
                # Charger l'image
                image = cv2.imread(os.path.join(folder_path, filename))

                # Appliquer l'effet miroir
                mirrored = cv2.flip(image, 1)

                # Ajouter du bruit sel et poivre (intensité de 0.05)
                noisy_sp = np.copy(image)
                # Générer un masque aléatoire pour le bruit sel et poivre
                sp_mask = np.random.rand(*image.shape[:2])
                noisy_sp[sp_mask < 0.025] = [0, 0, 0]  # Sel (5% des pixels deviennent noirs)
                noisy_sp[sp_mask > 0.975] = [255, 255, 255]  # Poivre (5% des pixels deviennent blancs)

                # Ajouter du bruit gaussien (écart-type de 0.1)
                noisy_gaussian = np.copy(image)
                gaussian = np.random.normal(0, 0.7, image.shape).astype(np.uint8)
                noisy_gaussian = cv2.add(image, gaussian)

                # Ajouter du bruit de poisson
                noisy_poisson = np.copy(image)
                noisy_poisson = np.random.poisson(image / 255.0 * 10) / 10 * 255

                # Sauvegarder les images augmentées dans le sous-répertoire correspondant
                cv2.imwrite(os.path.join(output_subdir, filename.split('.')[0] + '_mirrored.jpg'), mirrored)
                cv2.imwrite(os.path.join(output_subdir, filename.split('.')[0] + '_sp_noise.jpg'), noisy_sp)
                cv2.imwrite(os.path.join(output_subdir, filename.split('.')[0] + '_gaussian_noise.jpg'), noisy_gaussian)
                cv2.imwrite(os.path.join(output_subdir, filename.split('.')[0] + '_poisson_noise.jpg'), noisy_poisson)
