import requests

# URL de téléchargement de RAF-DB
url = "https://github.com/YuvalNirkin/face.evoLVe/blob/master/align/RafDB.mat?raw=true"

# Définissez le nom du fichier dans lequel vous souhaitez enregistrer la base de données
nom_fichier = "RafDB.mat"

# Effectuez la requête HTTP pour télécharger le fichier
response = requests.get(url)

# Vérifiez si la requête a réussi (code de statut 200)
if response.status_code == 200:
    # Enregistrez le contenu de la réponse dans le fichier
    with open(nom_fichier, 'wb') as fichier:
        fichier.write(response.content)
    print("Téléchargement réussi : RAF-DB a été enregistré dans", nom_fichier)
else:
    print("Erreur lors du téléchargement. Code de statut :", response.status_code)
