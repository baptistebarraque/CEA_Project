import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten, Concatenate,  Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import os
import re
from tensorflow.keras.optimizers import Adam


### Script permettant d'entraîner le modèle CNN sur les données de simulations 3D ###

def load_and_preprocess_data(data_folder, max_sequence_length=1000):
    """
    Charger et prétraiter les données des capteurs depuis un dossier contenant les fichiers
    au format numerocapteur_x_y_massexplosion
    
    Args:
        data_folder: Chemin vers le dossier contenant tous les fichiers de données
        max_sequence_length: Longueur maximale de la séquence temporelle
    
    Returns:
        X: Données des capteurs normalisées et formatées
        y: Coordonnées des sources d'explosion [x, y, z] (z=0 si non disponible)
    """
    # Lister tous les fichiers dans le dossier
    all_files = os.listdir(data_folder)
    
    # Regrouper les fichiers par simulation (x_y_massexplosion)
    simulations = {}
    pattern = r'(\d+)_(-?\d+\.?\d*)_(-?\d+\.?\d*)_(\d+\.?\d*)_(\d+\.?\d*)' 
    
    for file in all_files:
        match = re.search(pattern, file)
        if match:
            sensor_id = int(match.group(1))
            x_coord = float(match.group(2))
            y_coord = float(match.group(3))
            z_coord =float(match.group(4))
            mass = float(match.group(5))
            
            # Clé unique pour chaque simulation
            sim_key = f"{x_coord}_{y_coord}_{z_coord}_{mass}"
            
            if sim_key not in simulations:
                simulations[sim_key] = {
                    'files': [None] * 5,  # Un emplacement pour chaque capteur (1-5)
                    'coords': [x_coord, y_coord, z_coord]  # Z est défini à 0 si non disponible
                }
            
            # Stocker le chemin du fichier pour ce capteur dans cette simulation
            simulations[sim_key]['files'][sensor_id-1] = os.path.join(data_folder, file)
    
    # Afficher les simulations trouvées pour le débogage
    print(f"Nombre de simulations trouvées: {len(simulations)}")
    
    # Préparer les données pour l'entraînement
    all_sensor_data = []
    explosion_sources = []
    
    for sim_key, sim_data in simulations.items():
        # Vérifier que tous les capteurs sont disponibles pour cette simulation
        if None in sim_data['files']:
            print(f"Attention: Données manquantes pour certains capteurs dans la simulation {sim_key}")
            continue
        
        sensor_readings = []
        
        # Traiter les données de chaque capteur
        for sensor_file in sim_data['files']:
            try:
                df = pd.read_csv(sensor_file,skiprows=1, delim_whitespace=True, header=None, names=['time', 'pressure'])
                # Lire le fichier en ignorant la première ligne (commentaire)
                
                # Normaliser la pression
                scaler = MinMaxScaler()
                pressure_normalized = scaler.fit_transform(df['pressure'].values.reshape(-1, 1))
                
                # Padding ou troncature pour avoir une longueur fixe
                
                if len(pressure_normalized) > max_sequence_length:
                    pressure_normalized = pressure_normalized[:max_sequence_length]
                else:
                    padding = np.zeros((max_sequence_length - len(pressure_normalized), 1))
                    pressure_normalized = np.vstack([pressure_normalized, padding])
                
                sensor_readings.append(pressure_normalized)
            except Exception as e:
                print(f"Erreur lors du traitement du fichier {sensor_file}: {e}")
        
        # Vérifier que tous les capteurs ont des données
        if len(sensor_readings) == 5:
            all_sensor_data.append(np.hstack(sensor_readings))
            explosion_sources.append(sim_data['coords'][:2])
    
    X = np.array(all_sensor_data)
    y = np.array(explosion_sources)
    
    print(f"Dimensions finales des données: X shape = {X.shape}, y shape = {y.shape}")
    
    return X, y


### Nouveau modèle
def create_explosion_source_cnn(input_shape=(1000, 5)):
    """
    Crée un modèle CNN pour prédire les coordonnées de la source d'explosion
    
    Args:
        input_shape: Shape des données d'entrée (séquence temporelle, nombre de capteurs)
    
    Returns:
        Modèle Keras compilé
    """
    model = Sequential([
        # Première couche convolutive
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        
        # Deuxième couche convolutive
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Troisième couche convolutive
        Conv1D(256, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Aplatissement et couches denses
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        
        # Couche de sortie pour prédire x et y
        Dense(2, activation=None)  # Pas d'activation pour régression
    ])
    
    # Compilation du modèle
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


def train_explosion_source_model(X, y):
    """
    Entraîne le modèle CNN pour localiser la source d'explosion
    
    Args:
        X: Données d'entrée des capteurs (séquences de pression)
        y: Coordonnées des sources d'explosion
    
    Returns:
        Modèle entraîné et historique d'entraînement
    """
    # Séparation des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    
    # Normalisation des coordonnées cibles
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    
    # Création du modèle
    model = create_explosion_source_cnn()
    
    # Affichage du résumé du modèle
    model.summary()
    
    # Entraînement
    history = model.fit(
        X_train, 
        y_train_scaled,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=1
    )
    
    # Évaluation sur le jeu de test
    test_loss, test_mae = model.evaluate(
        X_test, 
        y_test_scaled
    )
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
    
    return model, history, y_scaler

if __name__=='__main__':
    X, y = load_and_preprocess_data('Simulations3D/')
    print(X)
        
        # Entraîner le modèle
    model, history, y_scaler = train_explosion_source_model(X, y)


    # Visualiser l'historique de perte
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Perte d\'entraînement')
    plt.plot(history.history['val_loss'], label='Perte de validation')
    plt.title('Évolution de la perte lors de l\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    plt.show()