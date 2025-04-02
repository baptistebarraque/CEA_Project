#%% IMPORTATIONS
import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft


###Script d'entrainement du modèle RandomForest
#%% FONCTIONS D'ANALYSE :
# Liste globale des features à utiliser :
FEATURES_TO_USE = [
    "first_pic",
    "amplitude_top",
    "Dt_pics",
    "energie",
    "temps_moyen_pics",
    "variance","frequence_dominante", "ecart_type", "skewness", "kurtosis"
]

# Fonction pour extraire les caractéristiques d'un signal
def trouver_pics(temps, points):
    # Identifie les pics dans le signal et retourne les indices et les valeurs des pics détectés.
    indices_pics, _ = find_peaks(points, distance=5, prominence=0.5)
    temps_pics = temps[indices_pics] if len(indices_pics) > 0 else [temps[-1] + 3]
    valeurs_pics = points[indices_pics] if len(indices_pics) > 0 else [10000]
    return temps_pics, valeurs_pics

def premier_pics(temps, points):
    # Renvoie le temps du premier pic détecté dans le signal.
    temps_pics, _ = trouver_pics(temps, points)
    return temps_pics[0]

def amplitude_maxi(temps, points):
    # Calcule l'amplitude maximale parmi les pics détectés.
    _, valeurs_pics = trouver_pics(temps, points)
    return max(valeurs_pics)

def Dt_premiers_pics(temps, points):
    # Calcule la différence de temps entre les deux premiers pics.
    temps_pics, _ = trouver_pics(temps, points)
    if len(temps_pics) < 2:
        return temps[-1] + 0.1
    return temps_pics[1] - temps_pics[0]

def energie_signal(points):
    # Calcule l'énergie totale du signal en sommant les carrés des valeurs du signal.
    return np.sum(np.square(points))

def temps_moyen_pics(temps, points):
    # Calcule le temps moyen entre les pics successifs.
    temps_pics, _ = trouver_pics(temps, points)
    if len(temps_pics) < 2:
        return 0
    return np.mean(np.diff(temps_pics))

def frequence_dominante(temps, points, echantillonnage=1.0):
    # Utilise la FFT pour identifier la fréquence dominante du signal.
    n = len(points)
    fft_vals = fft(points)
    freqs = np.fft.fftfreq(n, d=echantillonnage / n)
    dominant_freq = freqs[np.argmax(np.abs(fft_vals))]
    return np.abs(dominant_freq)

def variance_signal(points):
    # Calcule la variance du signal.
    return np.var(points)

def ecart_type_signal(points):
    # Calcule l'écart-type du signal.
    return np.std(points)

def skewness_signal(points):
    # Calcule l'asymétrie (skewness) du signal.
    return skew(points)

def kurtosis_signal(points):
    # Calcule l'aplatissement (kurtosis) du signal.
    return kurtosis(points)

def get_features(temps, points):
    # Compile toutes les caractéristiques calculées en un dictionnaire, puis les retourne sous forme de liste ordonnée.
    all_features = {
        "first_pic": premier_pics(temps, points),
        "amplitude_top": amplitude_maxi(temps, points),
        "Dt_pics": Dt_premiers_pics(temps, points),
        "energie": energie_signal(points),
        "temps_moyen_pics": temps_moyen_pics(temps, points),
        "variance": variance_signal(points),
        "frequence_dominante": frequence_dominante(temps, points),
        "ecart_type": ecart_type_signal(points),
        "skewness": skewness_signal(points),
        "kurtosis": kurtosis_signal(points)
    }

    return [all_features[feature] for feature in FEATURES_TO_USE]

def trouver_classe(chaine):
    # Extrait les paramètres de l'explosion (capteur, localisation, masse) à partir du nom de fichier.
    parts = chaine.split('_')
    capteur = int(parts[0])
    localisation = (float(parts[1]), float(parts[2]), float(parts[3]))
    masse = float(parts[4].replace('.txt', ''))
    return capteur, localisation, masse

#%% DICTIONNAIRE DE LA BASE DE DONNEES
# Load dataset
base_path = r"Simulations3D/"
os.chdir(base_path)
simulations = {}
for nom_doc in os.listdir():
    if '.txt' in nom_doc:
        data = np.loadtxt(nom_doc)
        temps, points = data[:, 0], data[:, 1]
        capteur, localisation, masse = trouver_classe(nom_doc)
        features = get_features(temps, points)
        key = (localisation, masse)
        if key not in simulations:
            simulations[key] = {}
        simulations[key][capteur] = features
print("Notre dictionnaire de base de données est prêt !")

#%% CREATION DU DATAFRAME
data_list = []
for (loc, masse), capteurs in simulations.items():
    features = []
    for capteur_id, valeurs in capteurs.items():
        features.extend(valeurs)  # Ajoute les features du capteur sans restriction

    # Ajoute les coordonnées et la masse
    features += [loc[0], loc[1], max(0, loc[2]), masse]
    data_list.append(features)

# Création dynamique des noms de colonnes
num_features_per_sensor = len(FEATURES_TO_USE)
columns = [f"feature_cap{capteur}_{feature}" for capteur in range(5) for feature in range(num_features_per_sensor)]
columns += ["x_localisation", "y_localisation", "z_localisation", "masse"]

# Vérification que les colonnes correspondent bien au nombre de features
max_columns = max(len(row) for row in data_list)
if len(columns) != max_columns:
    columns = [f"feature_{i}" for i in range(max_columns - 4)] + ["x_localisation", "y_localisation", "z_localisation", "masse"]

# Création du DataFrame
data_ML = pd.DataFrame(data_list, columns=columns)

print("Notre DataFrame est prêt!")

#%% ENTRAINEMENT DU MODELE
# Train model
X = data_ML.drop(["x_localisation", "y_localisation", "z_localisation", "masse"], axis=1)
Y_localisation = data_ML[["x_localisation", "y_localisation", "z_localisation"]]
Y_masse = data_ML["masse"]

X_train, X_test, Y_train_loc, Y_test_loc = train_test_split(X, Y_localisation, test_size=0.25, random_state=42)
X_train, X_test, Y_train_mass, Y_test_mass = train_test_split(X, Y_masse, test_size=0.25, random_state=42)

print("Entrainement du modèle en cours...")
#ajustement des paramètres pour le modèle de localisation
model_loc = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=700,       # Augmenter le nombre d'arbres
    max_depth=100,           # Limiter la profondeur des arbres
    min_samples_split=2,    # Nombre minimum d'échantillons pour diviser un nœud
    min_samples_leaf=1,     # Nombre minimum d'échantillons pour un nœud feuille
    max_features='sqrt',    # Nombre de caractéristiques à considérer
    bootstrap=False          # Échantillonnage avec remise
))

model_loc.fit(X_train, Y_train_loc)
Y_pred_loc = model_loc.predict(X_test)

# ajustement des paramètres pour le modèle de masse
model_mass = RandomForestRegressor(
    n_estimators=100,       # Augmenter le nombre d'arbres
    max_depth=20,           # Limiter la profondeur des arbres
    min_samples_split=2,    # Nombre minimum d'échantillons pour diviser un nœud
    min_samples_leaf=2,     # Nombre minimum d'échantillons pour un nœud feuille
    max_features='sqrt',    # Nombre de caractéristiques à considérer
    bootstrap=False          # Échantillonnage avec remise
)
model_mass.fit(X_train, Y_train_mass)
Y_pred_mass = model_mass.predict(X_test)

# Évaluation du modèle
distance_moyenne = np.mean(np.sqrt(np.sum((Y_test_loc - Y_pred_loc) ** 2, axis=1)))
mae_mass = mean_absolute_error(Y_test_mass, Y_pred_mass)

print(f"Nombre de situations d'entraînement : {len(X_train)}")
print(f"Nombre de situations de test : {len(X_test)}")
print("Distance moyenne localisation :", distance_moyenne)
print("Erreur Moyenne Absolue Masse :", mae_mass)

#%% FONCTION DE PREDICTION
def predict_from_sensors(dossier_capteurs):
    """Cette fonction prédit la localisation et la masse d'une explosion à partir des données de cinq capteurs.
    Entrée : Chemin d'un dossier contenant exactement cinq fichiers de capteurs au format .txt.
    Sortie : Localisation prédite (tuple de trois coordonnées) et masse prédite de l'explosion."""
    fichiers = [f for f in os.listdir(dossier_capteurs) if f.endswith('.txt')]
    if len(fichiers) != 5:
        raise ValueError("Le dossier doit contenir 5 fichiers de capteurs !")

    capteurs_data = {}
    for fichier in fichiers:
        data = np.loadtxt(os.path.join(dossier_capteurs, fichier))
        temps, points = data[:, 0], data[:, 1]
        capteur, _, _ = trouver_classe(fichier) # on connait pas le reste!
        capteurs_data[capteur] = get_features(temps, points)

    features = []
    for capteur_id in range(5):
        features.extend(capteurs_data.get(capteur_id, [0, 0, 0, 0]))

    import pandas as pd

    # Transformer le tableau de features en DataFrame avec les bons noms de colonnes
    features = np.array(features).reshape(1, -1)  # Conversion en numpy array + reshape
    features_df = pd.DataFrame(features, columns=X.columns)  # Création du DataFrame

    predicted_loc = model_loc.predict(features_df)[0]
    #predicted_mass = model_mass.predict(features_df)[0]

    return predicted_loc

# dossier = r"TE/"
# predict_from_sensors(dossier)
