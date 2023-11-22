import cv2
from datasets import load_dataset

dataset = load_dataset("Piro17/affectnethq")
if dataset == Null:
    print("erreur de chargement")
else :
    print("le chargement est ok")


class AffectNetDatabase:
    def __init__(self, directory):
        """
        Initialise la base de données AffectNet.

        :param directory: Chemin du répertoire contenant les images et les annotations.
        """
        self.directory = directory
        self.images = []  # Liste pour stocker les chemins des images
        self.annotations = []  # Liste pour stocker les annotations

    def load_data(self):
        """
        Charge les données de la base de données depuis le répertoire spécifié.
        """
        return

    def get_image(self, index):

        image_path = self.images[index]
        annotation = self.annotations[index]

        image = cv2.imread(image_path)  # Lecture de l'image

        return image, annotation

