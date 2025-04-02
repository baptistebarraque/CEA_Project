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

# Signal analysis functions
def load_signal_file(filepath):
    """Load a signal file and extract meaningful data"""
    try:
        data = np.loadtxt(filepath, comments='#')
        
        if len(data.shape) < 2 or data.shape[1] < 2:
            return None, None
            
        time = data[:, 0]
        signal = data[:, 1]
        
        # Find where the signal starts changing
        if np.all(signal == signal[0]):
            return time, signal
            
        changes = np.where(np.diff(signal) != 0)[0]
        if len(changes) > 0:
            start_idx = max(0, changes[0] - 10)
            return time[start_idx:], signal[start_idx:]
        
        return time, signal
    except Exception:
        return None, None

def normalize_min_max(signal):
    """Normalize signal to [0,1] range"""
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val == min_val:
        return np.zeros_like(signal)
    return (signal - min_val) / (max_val - min_val)

def preprocess_signals(signal1, signal2, normalize=True):
    """Preprocess two signals to make them comparable"""
    min_length = min(len(signal1), len(signal2))
    signal1 = signal1[:min_length]
    signal2 = signal2[:min_length]
    
    if normalize:
        signal1 = normalize_min_max(signal1)
        signal2 = normalize_min_max(signal2)
        
    return signal1, signal2

# Signal comparison metrics
def spectral_distance(signal1, signal2):
    try:
        signal1, signal2 = preprocess_signals(signal1, signal2)
        if len(signal1) < 10 or len(signal2) < 10:
            return float('inf')
        
        # Use smaller scales for better performance
        scales = np.arange(1, min(64, len(signal1) // 4))
        coeffs1, _ = pywt.cwt(signal1, scales, 'mexh')
        coeffs2, _ = pywt.cwt(signal2, scales, 'mexh')
        return np.sqrt(np.sum((coeffs1 - coeffs2)**2))
    except Exception:
        return float('inf')

def wasserstein_distance_fn(signal1, signal2):
    try:
        signal1, signal2 = preprocess_signals(signal1, signal2)
        if len(signal1) < 10 or len(signal2) < 10:
            return float('inf')
        
        # Downsample for performance
        if len(signal1) > 1000:
            step = len(signal1) // 1000
            signal1 = signal1[::step]
            signal2 = signal2[::step]
            
        return wasserstein_distance(signal1, signal2)
    except Exception:
        return float('inf')

def dtw_distance(signal1, signal2):
    try:
        signal1, signal2 = preprocess_signals(signal1, signal2)
        if len(signal1) < 10 or len(signal2) < 10:
            return float('inf')
        
        # Downsample for performance
        if len(signal1) > 500:
            step = len(signal1) // 500
            signal1 = signal1[::step]
            signal2 = signal2[::step]
            
        distance, _ = fastdtw(signal1, signal2)
        return distance
    except Exception:
        return float('inf')

def cross_correlation_distance(signal1, signal2):
    try:
        signal1, signal2 = preprocess_signals(signal1, signal2)
        if len(signal1) < 10 or len(signal2) < 10:
            return float('inf')
        
        # Use FFT-based correlation for speed
        correlation = correlate(signal1, signal2, mode='full', method='fft')
        max_corr = np.max(np.abs(correlation))
        return 1.0 / max_corr if max_corr != 0 else float('inf')
    except Exception:
        return float('inf')

def process_sensor(sensor_id, reference_signals, explosion_params, base_path, metrics):
    """
    Process all explosions for a specific sensor
    Returns: Dict of {explosion_name: {metric_name: distance}}
    """
    print(f"Processing sensor {sensor_id}...")
    results = {}
    
    # Skip if no reference signal for this sensor
    if sensor_id not in reference_signals:
        return results
        
    reference_signal = reference_signals[sensor_id]
    
    # Pre-load all signals for this sensor (caching)
    signal_cache = {}
    files = glob.glob(os.path.join(base_path, f"{sensor_id}_*"))
    total_files = len(files)
    
    for i, file_path in enumerate(files):
        if i % max(1, total_files // 5) == 0:
            progress = int(100 * i / total_files)
            print(f"  Sensor {sensor_id} loading: {progress}% ({i}/{total_files})")
            
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) >= 5:  # Need at least 5 parts for sensor_id, x, y, z, mass
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                mass = float(parts[4].split('.')[0])
                
                # Cache the signal
                position_key = f"x={x} y={y} z={z} m={mass}"
                time_data, signal = load_signal_file(file_path)
                
                if time_data is not None and signal is not None and len(signal) > 10 and np.std(signal) > 1e-10:
                    signal_cache[position_key] = signal
            except (ValueError, IndexError):
                pass
    
    # Process each explosion
    for params in explosion_params:
        x, y, z, mass = params
        explosion_name = f"x={x} y={y} z={z} m={mass}"
        
        # Skip if signal not in cache
        if explosion_name not in signal_cache:
            continue
            
        signal = signal_cache[explosion_name]
        
        # Calculate all metrics for this signal at once
        for metric_name, metric_fn in metrics.items():
            try:
                distance = metric_fn(reference_signal, signal)
                if distance != float('inf'):
                    if explosion_name not in results:
                        results[explosion_name] = {}
                    results[explosion_name][metric_name] = distance
            except Exception:
                pass
    
    print(f"Completed processing sensor {sensor_id}")
    return results


def run_analysis(reference_position, reference_mass, data_folder="donnees_simulation_3D"):
    """Run signal analysis and return top matches"""
    print(f"\n=== Running Analysis for reference position: ({reference_position[0]}, {reference_position[1]}, {reference_position[2]}), mass: {reference_mass} ===")
    
    num_sensors = 5  # Adjust based on your data
    start_time = time.time()
    
    # Base path for simulation data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, data_folder)
    
    print(f"Looking for simulation data in: {base_path}")
    if not os.path.exists(base_path):
        print(f"Error: Data folder not found at {base_path}")
        return []
    
    # Load reference signals - FIXED pattern to match {sensor_id}_x_y_z_mass.txt
    reference_signals = {}
    for sensor_id in range(0, num_sensors + 1):
        # Match reference pattern correctly for {sensor_id}_x_y_z_mass.txt
        ref_x, ref_y, ref_z = reference_position
        reference_pattern = f"{sensor_id}_{ref_x}_{ref_y}_{ref_z}_{reference_mass}"
        reference_files = glob.glob(os.path.join(base_path, f"{reference_pattern}*"))
        
        if reference_files:
            time_data, signal = load_signal_file(reference_files[0])
            if time_data is not None and signal is not None and np.std(signal) > 1e-10:
                reference_signals[sensor_id] = signal
                print(f"Loaded reference signal for sensor {sensor_id}: {reference_files[0]}")
    
    if not reference_signals:
        print("Error: No reference signals found. Check reference position and mass.")
        return []
    
    # Get all unique explosion parameters (excluding reference)
    all_files = glob.glob(os.path.join(base_path, "*"))
    explosion_params = set()
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) >= 5:  # Need at least 5 parts for sensor_id, x, y, z, mass
            try:
                sensor_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                mass = float(parts[4].split('.')[0])
                
                # Skip reference parameters
                if not (abs(x - reference_position[0]) < 0.01 and 
                        abs(y - reference_position[1]) < 0.01 and
                        abs(z - reference_position[2]) < 0.01 and
                        abs(mass - reference_mass) < 0.01):
                    explosion_params.add((x, y, z, mass))
            except (ValueError, IndexError):
                pass
    
    explosion_params = list(explosion_params)
    print(f"Found {len(explosion_params)} unique explosion simulations to compare")
    
    if not explosion_params:
        print("Error: No comparison explosion simulations found.")
        return []
    
    # Define metrics
    metric_functions = {
        'spectral': spectral_distance,
        'wasserstein': wasserstein_distance_fn,
        'dtw': dtw_distance,
        'cross_corr': cross_correlation_distance,
    }
    
    # Use multiprocessing to process sensors in parallel
    sensor_ids = [i for i in range(num_sensors + 1) if i in reference_signals]
    
    print(f"Processing {len(sensor_ids)} sensors for {len(explosion_params)} explosions in parallel...")
    
    # Process sensors in parallel using multiprocessing
    process_func = partial(
        process_sensor,
        reference_signals=reference_signals,
        explosion_params=explosion_params,
        base_path=base_path,
        metrics=metric_functions
    )
    
    # Use up to cpu_count() processes but not more than we have sensors
    num_processes = min(len(sensor_ids), cpu_count())
    print(f"Using {num_processes} parallel processes")
    
    with Pool(processes=num_processes) as pool:
        sensor_results = pool.map(process_func, sensor_ids)
    
    # Combine results from all sensors
    all_distances = {}
    for sensor_result in sensor_results:
        for explosion_name, metrics in sensor_result.items():
            if explosion_name not in all_distances:
                all_distances[explosion_name] = {}
                
            for metric_name, distance in metrics.items():
                if metric_name not in all_distances[explosion_name]:
                    all_distances[explosion_name][metric_name] = []
                all_distances[explosion_name][metric_name].append(distance)
    
    # Calculate average distances for each metric
    metric_distances = {metric: {} for metric in metric_functions}
    
    for explosion_name, metrics in all_distances.items():
        for metric_name, distances in metrics.items():
            if distances:
                metric_distances[metric_name][explosion_name] = np.mean(distances)
    
    # Sort by distance for each metric to get ranks
    explosion_metric_ranks = {}
    
    for metric_name, distances in metric_distances.items():
        sorted_explosions = sorted(distances.items(), key=lambda x: x[1])
        
        for rank, (explosion_name, _) in enumerate(sorted_explosions):
            if explosion_name not in explosion_metric_ranks:
                explosion_metric_ranks[explosion_name] = {}
            explosion_metric_ranks[explosion_name][metric_name] = rank
    
    # Calculate average ranks
    average_ranks = {}
    for explosion_name, metric_ranks in explosion_metric_ranks.items():
        if metric_ranks:
            avg_rank = sum(metric_ranks.values()) / len(metric_ranks)
            average_ranks[explosion_name] = avg_rank
    
    # Sort by average rank
    top_explosions = sorted(average_ranks.items(), key=lambda x: x[1])[:10]
    
    print(f"\n=== Top 10 Most Similar Explosions (By Average Rank) ===")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    
    for i, (explosion, rank) in enumerate(top_explosions):
        print(f"{i+1}. {explosion}: Average Rank = {rank:.2f}")
    
    # Extract coordinates and ranks for visualization - now including z
    top_positions = []
    for explosion_name, rank in top_explosions[:1]:
        parts = explosion_name.split()
        x = float(parts[0].split('=')[1])
        y = float(parts[1].split('=')[1])
        z = float(parts[2].split('=')[1])
        top_positions.append((x, y, z))
    
    return top_positions[0]