import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
import glob
import re

def parse_range_or_list(arg_str):
    """Parses a string like '0,1,5' or '0-5,10,12' into a set of integers."""
    if not arg_str:
        return None
    
    indices = set()
    parts = arg_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                if start > end:
                    raise ValueError(f"Invalid range: {start}-{end}")
                indices.update(range(start, end + 1))
            except ValueError as e:
                print(f"Warning: Could not parse range '{part}'. Skipping. Error: {e}")
        else:
            try:
                indices.add(int(part))
            except ValueError:
                print(f"Warning: Could not parse integer '{part}'. Skipping.")
    
    return sorted(list(indices)) if indices else None

def find_nearest_epoch(target_epoch, available_epochs):
    """Find the nearest available epoch to the target epoch."""
    if not available_epochs:
        return None
    return min(available_epochs, key=lambda x: abs(x - target_epoch))

def plot_model_probing_accuracy(results, model_run, epochs, output_dir, colors, label_map=None):
    """Plot probing accuracy for a specific model run and list of epochs."""
    if model_run not in results:
        print(f"Model run '{model_run}' not found in results")
        return None
    
    model_results = results[model_run]
    
    # Extract model type from the model_run name
    model_type = None
    if "base" in model_run.lower():
        model_type = "base"
    elif "conv" in model_run.lower():
        model_type = "conv"
    elif "fft" in model_run.lower():
        model_type = "fft"
    else:
        print(f"Could not determine model type from '{model_run}'")
        return None
    
    # Create figures dictionary to store one figure per epoch
    figures = {}
    
    for epoch in epochs:
        epoch_str = f"epoch_{epoch}"
        if epoch_str not in model_results:
            print(f"Epoch {epoch} not found for model {model_run}")
            continue
        
        # Create figure for this epoch
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Get baseline TF-IDF accuracy
        tfidf_accuracy = model_results.get('baseline_tfidf', {}).get('sentiment_probe', {}).get('accuracy', 0)
        
        # Find all layers for this epoch
        layers = []
        layer_initial_accuracies = []
        layer_trained_accuracies = []
        
        for key in model_results[epoch_str].keys():
            if key.startswith('layer_'):
                layer = int(key.split('_')[1])
                layers.append(layer)
                
                # Get initial state accuracy
                initial_acc = model_results[epoch_str][key].get('baseline_initial_state', {}).get('sentiment_probe', {}).get('accuracy', np.nan)
                layer_initial_accuracies.append(initial_acc)
                
                # Get trained state accuracy
                trained_acc = model_results[epoch_str][key].get('main_trained_state', {}).get('sentiment_probe', {}).get('accuracy', np.nan)
                layer_trained_accuracies.append(trained_acc)
        
        if not layers:
            print(f"No layer data found for {model_run} at epoch {epoch}")
            continue
        
        # Sort by layer number
        sorted_indices = np.argsort(layers)
        layers = np.array(layers)[sorted_indices]
        layer_initial_accuracies = np.array(layer_initial_accuracies)[sorted_indices]
        layer_trained_accuracies = np.array(layer_trained_accuracies)[sorted_indices]
        
        # Plot data
        line_width = 2.5
        marker_size = 8
        
        # Plot TFIDF accuracy (horizontal line)
        ax.axhline(y=tfidf_accuracy, color='gray', linestyle='-.', linewidth=line_width, alpha=0.7, label='TF-IDF Baseline')
        
        # Plot initial state accuracy
        ax.plot(layers, layer_initial_accuracies, color=colors[model_type]['light'], 
                linestyle='--', linewidth=line_width, marker='o', markersize=marker_size, 
                label='Initial State')
        
        # Plot trained state accuracy
        ax.plot(layers, layer_trained_accuracies, color=colors[model_type]['base'], 
                linestyle='-', linewidth=line_width, marker='s', markersize=marker_size, 
                label='Trained State')
        
        # Set plot properties
        ax.set_xlabel('Layer', fontsize=14)
        ax.set_ylabel('Probing Accuracy', fontsize=14)
        model_name = label_map.get(model_type, model_type.capitalize()) if label_map else model_type.capitalize()
        ax.set_title(f'{model_name} Model - Epoch {epoch}', fontsize=16)
        
        # Set x-axis to show integer ticks only
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Format y-axis to avoid scientific notation
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # Set y-axis limits to accommodate all models similarly
        ax.set_ylim(0.45, 0.85)
        
        # Add legend and grid
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Make plot tight
        plt.tight_layout()
        
        # Save the figure
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fig_path = os.path.join(output_dir, f"{model_run}_epoch_{epoch}_probing.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {fig_path}")
        
        # Store figure for later reference
        figures[epoch] = {'fig': fig, 'ax': ax, 'path': fig_path}
    
    return figures

def create_stage_visualization(results, model_runs, stage_epochs, output_dir, colors, label_map=None):
    """Create a 5x3 grid visualization for the different stages and models."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Setup the figure and grid
    fig = plt.figure(figsize=(18, 25))
    gs = gridspec.GridSpec(5, 3, figure=fig, wspace=0.25, hspace=0.4)
    
    # Stage titles
    stage_titles = [
        "Stage 0: Unstructured",
        "Stage 1: Emerging Structure",
        "Stage 2: Maximum Separation",
        "Stage 3: Morphological Changes",
        "Stage 4: Late Homogenization"
    ]
    
    # Column titles (model types)
    column_titles = ["Base Model", "Conv2D Model", "FFT Model"]
    
    # Process each stage and model
    for stage_idx, stage_title in enumerate(stage_titles):
        for model_idx, model_run in enumerate(model_runs):
            # Extract model type
            model_type = None
            if "base" in model_run.lower():
                model_type = "base"
            elif "conv" in model_run.lower():
                model_type = "conv"
            elif "fft" in model_run.lower():
                model_type = "fft"
            else:
                print(f"Could not determine model type from '{model_run}'")
                continue
            
            # Get the target epoch for this stage and model
            target_epoch = stage_epochs[stage_idx][model_idx]
            
            # Check if we have data for this epoch
            epoch_str = f"epoch_{target_epoch}"
            if model_run not in results or epoch_str not in results[model_run]:
                print(f"No data for {model_run} at epoch {target_epoch}")
                
                # Try to find nearest available epoch
                available_epochs = []
                if model_run in results:
                    for key in results[model_run].keys():
                        if key.startswith('epoch_'):
                            try:
                                available_epochs.append(int(key.split('_')[1]))
                            except ValueError:
                                continue
                
                if available_epochs:
                    nearest_epoch = find_nearest_epoch(target_epoch, available_epochs)
                    print(f"Using nearest available epoch {nearest_epoch} instead")
                    epoch_str = f"epoch_{nearest_epoch}"
                    target_epoch = nearest_epoch
                else:
                    print(f"No epochs available for {model_run}")
                    continue
            
            # Create subplot
            ax = fig.add_subplot(gs[stage_idx, model_idx])
            
            # Get baseline TF-IDF accuracy
            tfidf_accuracy = results[model_run].get('baseline_tfidf', {}).get('sentiment_probe', {}).get('accuracy', 0)
            
            # Find all layers for this epoch
            layers = []
            layer_initial_accuracies = []
            layer_trained_accuracies = []
            
            for key in results[model_run][epoch_str].keys():
                if key.startswith('layer_'):
                    layer = int(key.split('_')[1])
                    layers.append(layer)
                    
                    # Get initial state accuracy
                    initial_acc = results[model_run][epoch_str][key].get('baseline_initial_state', {}).get('sentiment_probe', {}).get('accuracy', np.nan)
                    layer_initial_accuracies.append(initial_acc)
                    
                    # Get trained state accuracy
                    trained_acc = results[model_run][epoch_str][key].get('main_trained_state', {}).get('sentiment_probe', {}).get('accuracy', np.nan)
                    layer_trained_accuracies.append(trained_acc)
            
            if not layers:
                print(f"No layer data found for {model_run} at epoch {target_epoch}")
                continue
            
            # Sort by layer number
            sorted_indices = np.argsort(layers)
            layers = np.array(layers)[sorted_indices]
            layer_initial_accuracies = np.array(layer_initial_accuracies)[sorted_indices]
            layer_trained_accuracies = np.array(layer_trained_accuracies)[sorted_indices]
            
            # Plot data
            line_width = 2.2
            marker_size = 6
            
            # Plot TFIDF accuracy (horizontal line)
            ax.axhline(y=tfidf_accuracy, color='gray', linestyle='-.', linewidth=1.8, alpha=0.7, label='TF-IDF')
            
            # Plot initial state accuracy
            ax.plot(layers, layer_initial_accuracies, color=colors[model_type]['light'], 
                    linestyle='--', linewidth=line_width, marker='o', markersize=marker_size, 
                    label='Initial')
            
            # Plot trained state accuracy
            ax.plot(layers, layer_trained_accuracies, color=colors[model_type]['base'], 
                    linestyle='-', linewidth=line_width, marker='s', markersize=marker_size, 
                    label='Trained')
            
            # Set plot properties
            ax.set_ylim(0.45, 0.85)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add title for first row only
            if stage_idx == 0:
                ax.set_title(column_titles[model_idx], fontsize=16)
            
            # Add epoch info to the plot
            ax.annotate(f"e{target_epoch}", xy=(0.05, 0.95), xycoords='axes fraction', 
                        fontsize=14, ha='left', va='top',
                        bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.7))
            
            # Add legend only to the first column
            if model_idx == 0:
                ax.legend(loc='upper center', fontsize=10, frameon=False)
            
            # Add y-label only to the first column
            if model_idx == 0:
                ax.set_ylabel('Probing Accuracy', fontsize=14)
            
            # Add x-label only to the last row
            if stage_idx == 4:
                ax.set_xlabel('Layer', fontsize=14)
            
            # Add stage labels to the left side
            if model_idx == 0:
                # Add text to the left of the plot
                fig.text(0.01, ax.get_position().y0 + ax.get_position().height/2, 
                         stage_title, fontsize=14, ha='left', va='center', rotation=90)
    
    # Add overall title
    plt.suptitle('Probing Accuracy Across Training Stages', fontsize=18, y=0.995)
    
    # Save the figure
    fig_path = os.path.join(output_dir, "stage_visualization_grid.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved grid visualization to {fig_path}")
    
    return fig_path

def main():
    parser = argparse.ArgumentParser(description="Generate probing accuracy visualizations.")
    parser.add_argument("--results_file", type=str, required=True, 
                        help="Path to the JSON file with probing results.")
    parser.add_argument("--epochs", type=str, required=True,
                        help="Comma-separated list/ranges of epochs to visualize.")
    parser.add_argument("--output_dir", type=str, default="./probing_viz",
                        help="Directory to save visualization outputs.")
    parser.add_argument("--model_runs", type=str, nargs='+', default=None,
                        help="List of model run names to include in visualization.")
    
    args = parser.parse_args()
    
    # Parse epochs
    epochs_to_viz = parse_range_or_list(args.epochs)
    if not epochs_to_viz:
        print("Error: No valid epochs specified.")
        return
    
    # Load results
    try:
        with open(args.results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading results file: {e}")
        return
    
    # If no specific model runs provided, use all in the results
    if args.model_runs is None:
        args.model_runs = list(results.keys())
    else:
        # Validate model runs
        for model_run in args.model_runs:
            if model_run not in results:
                print(f"Warning: Model run '{model_run}' not found in results. Skipping.")
        args.model_runs = [m for m in args.model_runs if m in results]
    
    if not args.model_runs:
        print("Error: No valid model runs found.")
        return
    
    # Define colors for each model type
    colors = {
        "base": {"base": "#2A4F74", "light": "#6A8FB4"},  # Dark and light steel blue
        "conv": {"base": "#1C5B3A", "light": "#5C9B7A"},  # Dark and light sea green
        "fft": {"base": "#CC594E", "light": "#EC898E"}    # Dark and light coral pink
    }
    
    # Map model types to display names
    label_map = {
        "base": "Base",
        "conv": "Conv2D",
        "fft": "FFT"
    }
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Generate individual model visualizations
    for model_run in args.model_runs:
        plot_model_probing_accuracy(results, model_run, epochs_to_viz, args.output_dir, colors, label_map)
    
    # Define stage epochs for each model [Base, Conv2D, FFT]
    stage_epochs = [
        [0, 0, 0],      # Stage 0
        [1, 1, 4],      # Stage 1
        [3, 5, 7],      # Stage 2
        [5, 6, 10],     # Stage 3
        [6, 9, 13]      # Stage 4
    ]
    
    # Create the 5x3 grid visualization
    create_stage_visualization(results, args.model_runs, stage_epochs, args.output_dir, colors, label_map)
    
    print("Visualization generation complete.")

if __name__ == "__main__":
    main()