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

# Suppress warnings
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")


### Script de visualisation des résultats en 3D ###



def visualize_city_3d_with_analysis(reference_position, reference_mass, top_matches=None, terrain_file="topoRelief.dat"):
    """
    Visualize city with added analysis results
    reference_position: (x, y, z) tuple
    reference_mass: float
    top_matches: list of (x, y, z, score) tuples
    """
    # Use absolute path if terrain_file is not absolute
    if not os.path.isabs(terrain_file):
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Join it with the terrain_file name
        terrain_file = os.path.join(script_dir, terrain_file)
    
    print(f"Loading terrain data from: {terrain_file}")
    print(f"File exists: {os.path.exists(terrain_file)}")
    
    # Define buildings with their coordinates and heights
    buildings = {
        "Bat1": {"coords": [(10.08, 10.223), (30.08, 10.223), (30.08, 15.223), (17.08, 15.223), (17.08, 20.223), (10.08, 25.223)], "height": 1.0},
        "Bat2": {"coords": [(50.08, 12.223), (65.08, 12.223), (65.08, 27.223), (50.08, 27.223)], "height": 2.0},
        "Bat3": {"coords": [(80.08, 32.223), (87.08, 32.223), (87.08, 57.223), (80.08, 57.223)], "height": 2.0},
        "Bat4": {"coords": [(12.08, 78.223), (42.08, 78.223), (42.08, 92.223), (12.08, 92.223)], "height": 3.0},
        "Bat5": {"coords": [(75, 70), (85, 80), (90, 75), (96, 81), (83.5, 83.5), (62.5, 62.5)], "height": 2.0},
        "Bat6": {"coords": [(20.08, 40.223), (40.08, 40.223), (40.08, 45.223), (25.08, 45.223), (25.08, 55.223), 
                (35.08, 55.223), (35.08, 70.223), (20.08, 70.223)], "height": 2.5},
    }
        
    # Load terrain data
    x_data, y_data, z_data = [], [], []
    with open(terrain_file, 'r') as f:
        for line in f:
            if line.strip().startswith('//'):  # Skip comments
                continue
            
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    x_data.append(float(parts[0]))
                    y_data.append(float(parts[1]))
                    z_data.append(float(parts[2]))
                except ValueError:
                    continue

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    z_data = np.array(z_data)
    
    # Create a grid for interpolation
    grid_size = 100
    xi = np.linspace(min(x_data), max(x_data), grid_size)
    yi = np.linspace(min(y_data), max(y_data), grid_size)
    X, Y = np.meshgrid(xi, yi)
    
    # Interpolate Z values for the terrain surface
    Z = griddata((x_data, y_data), z_data, (X, Y), method='linear')
    
    # Create 3D plot with improved settings
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 0.4])  # Adjust for better depth perception
# Remove grid, panes and background color for cleaner visualization
    ax.grid(False)  # Remove the grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none') 
    # Plot terrain surface with improved settings
    terrain = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.9, edgecolor=None, 
                            rstride=2, cstride=2, zorder=1)
    
   # Plot buildings
    colors = plt.cm.tab10(np.linspace(0, 1, len(buildings)))
    for (name, building), color in zip(buildings.items(), colors):
        # Extract building coordinates
        coords = building["coords"]
        height = building["height"]  # Get custom height for this building
        
        x_building = [p[0] for p in coords]
        y_building = [p[1] for p in coords]
        
        # Get terrain heights at building positions
        z_terrain = griddata((x_data, y_data), z_data, (x_building, y_building), method='linear')
        
        # Create building walls
        for i in range(len(coords)):
            j = (i + 1) % len(coords)
            # Create a vertical face for each side of the building
            x_face = [x_building[i], x_building[j], x_building[j], x_building[i]]
            y_face = [y_building[i], y_building[j], y_building[j], y_building[i]]
            z_face = [z_terrain[i], z_terrain[j], z_terrain[j] + height, z_terrain[i] + height]
            
            # Create vertical side with higher zorder
            vertices = [list(zip(x_face, y_face, z_face))]
            poly = Poly3DCollection(vertices, alpha=1)  # Slightly transparent
            poly.set_facecolor(color)
            poly.set_edgecolor('black')
            poly.set_zorder(10)  # Higher zorder to ensure visibility
            ax.add_collection3d(poly)
        
        # Create building top/roof with highest zorder
        top_verts = [list(zip(x_building, y_building, z_terrain + height))]
        top_poly = Poly3DCollection(top_verts, alpha=1)
        top_poly.set_facecolor(color)
        top_poly.set_edgecolor('black')
        top_poly.set_zorder(15)  # Even higher zorder
        ax.add_collection3d(top_poly)
        
        # Create building top/roof
        top_verts = [list(zip(x_building, y_building, z_terrain + height))]
        top_poly = Poly3DCollection(top_verts, alpha=1.0)
        top_poly.set_facecolor(color)
        top_poly.set_edgecolor('black')
        ax.add_collection3d(top_poly)
    
      # Add analysis visualization with improved visibility
    ref_x, ref_y, ref_z = reference_position
    # Make reference point larger and ensure visibility
    ax.scatter([ref_x], [ref_y], [ref_z], color='red', s=150, marker='*', 
             edgecolors='black', linewidth=1.5, zorder=100,
             label=f'Reference ({ref_x:.1f}, {ref_y:.1f}, {ref_z:.1f}, mass={reference_mass})')
    colors = ['red', 'blue', 'green']
    i=0
    for match in top_matches:
        match_x, match_y, match_z, model=match
        ax.scatter([match_x], [match_y], [match_z ], color=colors[i], s=50, marker='o', 
             edgecolors='black', linewidth=1.5, zorder=100,
             label=f'{model}: ({match_x:.1f}, {match_y:.1f}, {match_z:.1f}, mass={reference_mass})')
        i+=1

    
    # If we have analysis results, plot them
    # if top_matches:
    #     # Create color gradient for top matches - FIXED color handling
    #     match_colors = plt.cm.YlOrRd_r(np.linspace(0, 0.8, len(top_matches)))
        
    #     # Plot each top match with improved visibility
    #     for i, (match_x, match_y, match_z, score) in enumerate(top_matches):
    #         match_color = tuple(match_colors[i])
            
    #         # Increase marker size and add edge color
    #         ax.scatter([match_x], [match_y], [match_z + 3], color=match_color, s=120, 
    #                  marker='o', edgecolors='black', linewidth=1.5, zorder=90,
    #                  label=f'Match {i+1} (score: {score:.2f})')
            
    #         # Make connecting lines more visible
    #         ax.plot([ref_x, match_x], [ref_y, match_y], [ref_z + 3, match_z + 3], 
    #                color=match_color, linestyle='--', linewidth=2.5, alpha=0.8, zorder=80)
        
    #     # If we have more than one match, calculate and plot barycenter
    #     if len(top_matches) > 1:
    #         x_coords = [x for x, y, z, _ in top_matches]
    #         y_coords = [y for x, y, z, _ in top_matches]
    #         z_coords = [z for x, y, z, _ in top_matches]
    #         barycenter = (np.mean(x_coords), np.mean(y_coords), np.mean(z_coords))
            
    #         # Plot barycenter with clear visibility
    #         bary_color = tuple(plt.cm.Greens(0.6))
    #         ax.scatter([barycenter[0]], [barycenter[1]], [barycenter[2] + 3], 
    #                  color=bary_color, s=200, marker='D', edgecolors='black', linewidth=1.5,
    #                  zorder=95, label=f'Barycenter ({barycenter[0]:.1f}, {barycenter[1]:.1f}, {barycenter[2]:.1f})')
            
    #         # Make connecting line to barycenter more visible
    #         ax.plot([ref_x, barycenter[0]], [ref_y, barycenter[1]], [ref_z + 3, barycenter[2] + 3], 
    #                color=bary_color, linestyle='-', linewidth=3, alpha=0.9, zorder=85)
            
     # Improve camera position and view
    ax.view_init(elev=40, azim=225)  # Set a better default viewing angle
    try:
        ax.dist = 14  # Default is usually around 8-10
    except:
        pass  # Fallback for older Matplotlib versions
    # Set z-limits to ensure all points are visible
    z_min = np.min(Z)
    z_max = np.max([point[2] + 5 for point in top_matches]) if top_matches else np.max(Z) + 5
    ax.set_zlim(z_min, z_max)
    
    # Set plot properties
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('3D City Visualization with Explosion Analysis')
    
    # Set axis limits
    ax.set_xlim(min(x_data), max(x_data))
    ax.set_ylim(min(y_data), max(y_data))
    
    # Add legend for buildings and analysis points
    from matplotlib.patches import Patch
    building_handles = [Patch(color=color, label=name) for (name, _), color in zip(buildings.items(), colors)]
    # Combine all handles for legend
    handles, labels = ax.get_legend_handles_labels()
    all_handles = building_handles + handles
    all_labels = [h.get_label() for h in building_handles] + labels
    
    ax.legend(all_handles, all_labels, loc='upper right')
    plt.figtext(0.5, 0.01, "Click and drag to rotate the view. Use scroll wheel to zoom.", 
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    return fig, ax
