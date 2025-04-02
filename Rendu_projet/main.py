import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
import os
from scipy.stats import wasserstein_distance
from scipy.signal import correlate
from fastdtw import fastdtw
import pywt
import glob
import time
import logging
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import joblib
import distances_algorithm
import cnn_algorithm
import visualization
#import ML_RandomForest3D
from cnn_algorithm import *
from visualization import *
from distances_algorithm import *
from parallelization3D import find_closest_z
from parallelization3D import load_topography_data
from visualization import *
import joblib
from ML_RandomForest3D import *

###Script final###
# Suppress warnings
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

file_path = 'topoRelief.dat'

# Charger les données de topographie
topo_data = load_topography_data(file_path)

cnn_model=tf.keras.models.load_model('cnn_model.h5')
RF_model=joblib.load('C:/Users/bapti/Documents/Centrale vf/High Performance Calculus/Projet ST7/Rendu_projet/random_forest_localisation_model.pkl')


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

    predicted_loc = RF_model.predict(features_df)[0]
    #predicted_mass = model_mass.predict(features_df)[0]

    return predicted_loc

def main():
    matches=[]
    X,y =load_and_preprocess_data('Simulations3D')
    X_try,y_try=load_and_preprocess_data('TE/')
    print(X_try)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_try_scaled=y_scaler.transform(y_try)
    cnn_prediction=cnn_model.predict(X_try)
    y_pred_original = y_scaler.inverse_transform(cnn_prediction)
    y_pred_original=y_pred_original[0]
    x_cnn,y_cnn=y_pred_original
    z_cnn=find_closest_z(topo_data,x_cnn,y_cnn)
    matches.append((x_cnn,y_cnn,z_cnn,'CNN_Model'))
    distance_sol=run_analysis((51.3,51.3,9.2),40.0,"Simulations3D")
    x_dist,y_dist,z_dist=distance_sol
    print(x_dist,y_dist,z_dist)
    matches.append((x_dist,y_dist,z_dist,'Distance Model'))
    RF_predicted_coordinates=RF_model.predict(X_try)
    x_RF,y_RF,z_RF=predict_from_sensors('TE')
    matches.append((x_RF,y_RF,z_RF,'Random Forest Model'))
    reference_position = (51.3,51.3,9.2)  
    reference_mass = 40.0
    visualize_city_3d_with_analysis(reference_position,reference_mass,matches,'topoRelief.dat')

if __name__=='__main__':
    main()