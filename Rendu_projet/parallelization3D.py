import os
import subprocess
from itertools import product
import numpy as np
import matplotlib.path as mpath


### Script permettant de lancer des simulations en parallèle pour le cas 3D ###

def point_in_polygon(point, polygon):
    """
    Vérifie si un point est à l'intérieur d'un polygone en utilisant matplotlib.path.Path.

    :param point: Le point à tester (x, y)
    :param polygon: Liste des points du polygone [(x1, y1), (x2, y2), ...]
    :return: True si le point est dans le polygone, False sinon
    """
    path = mpath.Path(polygon)  # Création du polygone à partir de matplotlib.path.Path
    return path.contains_point(point)


def generate_grid(xmin, xmax, ymin, ymax, step, buildings):
    """
    Génère un cadrillage de points dans une zone délimitée, en évitant les bâtiments.

    :param xmin: Coordonnée x minimale de la zone
    :param xmax: Coordonnée x maximale de la zone
    :param ymin: Coordonnée y minimale de la zone
    :param ymax: Coordonnée y maximale de la zone
    :param step: Pas du cadrillage
    :param buildings: Liste de bâtiments, chaque bâtiment étant une liste de coordonnées [(x1, y1), (x2, y2), ...]
    :return: Liste des points valides du cadrillage
    """
    grid_points = []

    # Création des polygones représentant les bâtiments


    # Génération des points du cadrillage
    x_values = np.arange(xmin + step, xmax, step)
    y_values = np.arange(ymin + step, ymax, step)

    for x in x_values:
        for y in y_values:
            valid=True
            for building in buildings:

                if point_in_polygon((x,y),building):
                    valid=False
                    break
            if valid:
                grid_points.append((x, y))

    return grid_points
# LE BATI

    ####### Batiment 1
x0 = 10.08 ; y0 = 10.223
x1 = 30.08 ; y1 = 10.223
x2 = 30.08 ; y2 = 15.223
x3 = 17.08 ; y3 = 15.223
x4 = 17.08 ; y4 = 25.223
x5 = 10.08 ; y5 = 25.223
Bat1    = [ (x0, y0), (x1, y1) , (x2, y2), (x3,y3), (x4, y4), (x5, y5) ]




    ####### Batiment 2
x0 = 50.08 ; y0 = 12.223
x1 = 65.08 ; y1 = 12.223
x2 = 65.08 ; y2 = 27.223
x3 = 50.08 ; y3 = 27.223
Bat2    = [ (x0, y0), (x1, y1) , (x2, y2), (x3,y3) ]



    ####### Batiment 3
x0 = 80.08 ; y0 = 32.223
x1 = 87.08 ; y1 = 32.223
x2 = 87.08 ; y2 = 57.223
x3 = 80.08 ; y3 = 57.223

Bat3    = [ (x0, y0), (x1, y1) , (x2, y2), (x3,y3) ]



    ####### Batiment 4
x0 = 12.08 ; y0 = 78.223
x1 = 42.08 ; y1 = 78.223
x2 = 42.08 ; y2 = 92.223
x3 = 12.08 ; y3 = 92.223

Bat4    = [ (x0, y0), (x1, y1) , (x2, y2), (x3,y3) ]




    ####### Batiment 5
Bx0 = 75   ; y0 = 70
x1 = 85   ; y1 = 80
x2 = 90   ; y2 = 75
x3 = 96   ; y3 = 81
x4 = 0.5*167 ; y4 = 0.5*187
x5 = 0.5*135 ; y5 = 0.5*155

Bat5    = [ (x0, y0), (x1, y1) , (x2, y2), (x3,y3), (x4, y4), (x5, y5) ]





    ####### Batiment 6
x0 = 20.08 ; y0 = 40.223
x1 = 40.08 ; y1 = 40.223
x2 = 40.08 ; y2 = 45.223
x3 = 25.08 ; y3 = 45.223
x4 = 25.08 ; y4 = 55.223
x5 = 35.08 ; y5 = 55.223
x6 = 35.08 ; y6 = 70.223
x7 = 20.08 ; y7 = 70.223

Bat6    = [ (x0, y0), (x1, y1) , (x2, y2), (x3,y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7) ]

buildings=[Bat1,Bat2,Bat3,Bat4,Bat5,Bat6]
positions=generate_grid(0,100,0,100,5,buildings)
M_TNT_Values=[40]
simulated_positions=[]


# Générer toutes les combinaisons de paramètres
def load_topography_data(file_path):
    # Charger les données de topographie depuis le fichier
    data = np.loadtxt(file_path)
    return data

def find_closest_z(topo_data, x, y):
    # Calculer la distance entre le point (x, y) et chaque point dans les données
    distances = np.sqrt((topo_data[:, 0] - x)**2 + (topo_data[:, 1] - y)**2)

    # Trouver l'index du point le plus proche
    closest_index = np.argmin(distances)

    # Retourner la valeur Z du point le plus proche
    return topo_data[closest_index, 2]

def get_z_coordinates(topo_data, coordinates):
    z_values = []
    for (x, y) in coordinates:
        z = find_closest_z(topo_data, x, y)
        z_values.append(z)
    return z_values

# Chemin vers le fichier topoRelief.dat
#file_path = 'topoRelief.dat'

# Charger les données de topographie
#topo_data = load_topography_data(file_path)

# Liste de coordonnées (X, Y) pour lesquelles vous souhaitez obtenir les Z
def divide_list_into_sublists(original_list, num_sublists):
    # Calculer la taille de chaque sous-liste
    sublist_size = len(original_list) // num_sublists
    remainder = len(original_list) % num_sublists

    # Initialiser les sous-listes
    sublists = [[] for _ in range(num_sublists)]

    # Remplir les sous-listes
    index = 0
    for i in range(num_sublists):
        # Ajouter les éléments principaux
        sublists[i].extend(original_list[index:index + sublist_size])
        index += sublist_size

        # Ajouter un élément supplémentaire si nécessaire pour gérer le reste
        if remainder > 0:
            sublists[i].append(original_list[index])
            index += 1
            remainder -= 1

    return sublists

script_path = os.path.join(os.getcwd(), "SouffleBati3D.py")
sublist_positions=divide_list_into_sublists(positions, 20)
print(len(sublist_positions[0]))


def run_simulation(XC, YC,ZC, M_TNT):
    print(f"Lancement de la simulation avec XC={XC}, YC={YC},ZC={ZC}, M_TNT={M_TNT}")
    subprocess.Popen([
        "python3", script_path,
        str(XC),
        str(YC),
        str(ZC),
        str(M_TNT)])
# for XC, YC in sublist_positions[0]:
#     ZC=find_closest_z(topo_data, XC, YC)
#     for M_TNT in M_TNT_Values:
#         if (XC, YC, ZC, M_TNT) not in simulated_positions:
#             run_simulation(XC, YC, ZC, M_TNT)

def refinement(localisation, step, buildings):
    x,y=localisation
    M_TNT='40_refinement'
    grid=[(x+step,y+step),(x+step,y),(x+step,y-step),(x,y+step),(x,y-step),(x,y),(x-step,y+step),(x-step,y),(x-step,y-step)]
    for building in buildings:
        for position in grid:
            if point_in_polygon(position,building):
                grid.pop(position)
    print('Refinement in process, currently running',len(grid),'simulations')
    for position in grid:
        z=find_closest_z(topo_data,x,y)
        run_simulation(x,y,z,M_TNT)