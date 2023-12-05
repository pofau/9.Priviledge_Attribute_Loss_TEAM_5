import shutil
import os

# Définir le chemin vers le fichier texte contenant les labels
labels_file = r"C:\Users\lenov\Desktop\Cours Sorbonne\MLA\Projet\RAF-DB/list_partition_label.txt"

with open(labels_file, 'r') as file:
    lines = file.readlines()

data_list = [(line.split()[0], int(line.split()[1])) for line in lines]

for image, label in data_list:

    # Définir le chemin vers chaque image
    src = r"C:\Users\lenov\Desktop\Cours Sorbonne\MLA\Projet\RAF-DB\Image\{}".format(image)

    # Définir le chemin de distination
    dest = r"C:\Users\lenov\Desktop\Cours Sorbonne\MLA\Projet\RAF-DB\Classified_data\{}\{}".format(label,image)
    
    shutil.copy2(src, dest)