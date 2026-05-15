# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 23:54:22 2026

@author: Anja Ny Aina Harisoa
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

def plot_robots_and_stations(robot_points, station_X, station_Y, sol_n, sol_a, sol_h,
                              robot_ranges=None, range_max=175,
                              title='Robot & Charging Station Assignment',
                              xlabel='X (km)', ylabel='Y (km)', save_path=None):

    robot_x, robot_y = zip(*robot_points)
    I = len(robot_points)
    J = len(station_X)

    rect_w = 300
    rect_h = 200

    fig, ax = plt.subplots(figsize=(13, 9))

    # --- Draw assignment lines ---
    for i in range(I):
        for j in range(J):
            if sol_a.get((i, j), 0) == 1:
                ax.plot([robot_x[i], station_X[j]], [robot_y[i], station_Y[j]],
                        color='steelblue', linewidth=1.2, alpha=0.7, zorder=1)
            elif sol_h.get((i, j), 0) == 1:
                ax.plot([robot_x[i], station_X[j]], [robot_y[i], station_Y[j]],
                        color='steelblue', linewidth=1.2, alpha=0.7, zorder=1, linestyle='--')

    # --- Draw stations as rectangles ---
    for j in range(J):
        sx, sy = station_X[j], station_Y[j]
        rect = mpatches.FancyBboxPatch(
            (sx - rect_w / 2, sy - rect_h / 2), rect_w, rect_h,
            boxstyle="round,pad=20",
            linewidth=2, edgecolor='darkorange', facecolor='#FFF3E0', zorder=3
        )
        # --- Station fill colour based on number of assigned robots ---
        station_assigned = [sum(sol_a.get((i,j), 0) + sol_h.get((i,j), 0) for i in range(I)) for j in range(J)]
        ax.add_patch(rect)
        ax.text(sx, sy + rect_h / 2 + 60, f'S{j+1}\n({int(sol_n[j])} chargers)\n{station_assigned[j]} robots',
        ha='center', va='bottom', fontsize=8.5,
        fontweight='bold', color='darkorange', zorder=4)

    # --- Draw robots as points (colored by range) ---
    cmap = mcolors.LinearSegmentedColormap.from_list('range_cmap', ['yellow', 'green'])
    norm = mcolors.Normalize(vmin=0, vmax=range_max)
    sc = ax.scatter(robot_x, robot_y, c=robot_ranges, cmap=cmap, norm=norm,
                    s=100, zorder=5, edgecolors='white', linewidths=1.5)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Range (km)', fontsize=11)
    cbar.set_ticks([0, 50, 100, 150, range_max])

    for i, (xi, yi) in enumerate(zip(robot_x, robot_y)):
        ax.annotate(f'R{i+1}', (xi, yi),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7.5, color='dimgray', zorder=6)

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor='#FFF3E0', edgecolor='darkorange', label='Charging Station'),
        plt.Line2D([0], [0], color='steelblue', linewidth=1.5, label='Robot moves itself'),
        plt.Line2D([0], [0], color='steelblue',   linewidth=1.5, linestyle='--', label='Transported by human'),
        mpatches.Patch(facecolor='yellow', edgecolor='gray', label='Low range'),
        mpatches.Patch(facecolor='green',  edgecolor='gray', label='High range'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
              framealpha=0.9, edgecolor='gray')

    # --- Total robots label (outside plot) ---
    fig.text(0.99, 0.99, f'Total robots: {I}   |   Total stations: {J}',
             fontsize=10, fontweight='bold', color='steelblue',
             ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='steelblue', alpha=0.8))

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")

    plt.show()

def plot_robots_with_range(sample_data, title=None, save_path=None):
    """
    Plot robots as points colored by their range (red=low, blue=high).

    Parameters:
        sample_data : DataFrame with columns ['longitude', 'latitude', 'range']
        title       : plot title (auto-generated if None)
        save_path   : optional path to save the figure
    """
    subset_size = len(sample_data)
    if title is None:
        title = f'Robot Positions — Sample size {subset_size}'

    # Colormap: red (low range) → blue (high range)
    cmap = mcolors.LinearSegmentedColormap.from_list('range_cmap', ['red', 'blue', 'green'])
    norm = mcolors.Normalize(vmin=0, vmax=175)

    fig, ax = plt.subplots(figsize=(12, 8))

    sc = ax.scatter(
        sample_data['longitude'], sample_data['latitude'],
        c=sample_data['range'], cmap=cmap, norm=norm,
        s=100, zorder=5, edgecolors='white', linewidths=1.2
    )

    # Robot labels
    for _, row in sample_data.iterrows():
        ax.annotate(f"R{int(row['index'])}",
                    (row['longitude'], row['latitude']),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color='dimgray', zorder=6)

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Range (km)', fontsize=11)
    cbar.set_ticks([0, 50, 100, 150, 175])

    # Total robots outside plot
    fig.text(0.99, 0.99, f'Total robots: {subset_size}',
             fontsize=10, fontweight='bold', color='steelblue',
             ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='steelblue', alpha=0.8))

    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")

    plt.show()

#Show how busy stations are
def plot_robots_and_stations_states(robot_points, station_X, station_Y, sol_n, sol_a, sol_h,
                              robot_ranges=None, range_max=175,
                              title='Robot & Charging Station Assignment',
                              xlabel='X (km)', ylabel='Y (km)', save_path=None):

    robot_x, robot_y = zip(*robot_points)
    I = len(robot_points)
    J = len(station_X)

    rect_w = 300
    rect_h = 200

    # --- Station colours (up to 20 stations) ---
    station_cmap = plt.cm.get_cmap('tab20', 20)
    station_colors = [station_cmap(j) for j in range(J)]

    # --- Station fill colour based on number of assigned robots ---
    station_assigned = [sum(sol_a.get((i,j), 0) + sol_h.get((i,j), 0) for i in range(I)) for j in range(J)]
    fill_cmap = mcolors.LinearSegmentedColormap.from_list('fill_cmap', ['yellow', 'red'])
    fill_norm = mcolors.Normalize(vmin=0, vmax=max(station_assigned) if max(station_assigned) > 0 else 1)
    station_fill_colors = [fill_cmap(fill_norm(n)) for n in station_assigned]

    fig, ax = plt.subplots(figsize=(13, 9))

    # --- Draw assignment lines ---
    for i in range(I):
        for j in range(J):
            color = station_colors[j]
            if sol_a.get((i, j), 0) == 1:
                ax.plot([robot_x[i], station_X[j]], [robot_y[i], station_Y[j]],
                        color=color, linewidth=1.2, alpha=0.7, zorder=1)
            elif sol_h.get((i, j), 0) == 1:
                ax.plot([robot_x[i], station_X[j]], [robot_y[i], station_Y[j]],
                        color=color, linewidth=1.2, alpha=0.7, zorder=1, linestyle='--')

    # --- Draw stations as rectangles ---
    for j in range(J):
        sx, sy = station_X[j], station_Y[j]
        color = station_colors[j]
        rect = mpatches.FancyBboxPatch(
            (sx - rect_w / 2, sy - rect_h / 2), rect_w, rect_h,
            boxstyle="round,pad=20",
            linewidth=2, edgecolor=color, facecolor=station_fill_colors[j], zorder=3
        )
        ax.add_patch(rect)
        ax.text(sx, sy + rect_h / 2 + 60, f'S{j+1}\n({int(sol_n[j])} chargers)\n{station_assigned[j]} robots',
        ha='center', va='bottom', fontsize=8.5,
        fontweight='bold', color=color, zorder=4)

    # --- Draw robots as points (colored by range, edged by station) ---
    cmap = mcolors.LinearSegmentedColormap.from_list('range_cmap', ['yellow', 'green'])
    norm = mcolors.Normalize(vmin=0, vmax=range_max)

    for i, (xi, yi) in enumerate(zip(robot_x, robot_y)):
        edge_color = 'white'  # default if unassigned
        for j in range(J):
            if sol_a.get((i, j), 0) == 1 or sol_h.get((i, j), 0) == 1:
                edge_color = station_colors[j]
                break
        range_val = robot_ranges[i] if robot_ranges is not None else 0
        color = cmap(norm(range_val))
        ax.scatter(xi, yi, color=color, s=100, zorder=5,
                   edgecolors=edge_color, linewidths=2.5)
        ax.annotate(f'R{i+1}', (xi, yi),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7.5, color='dimgray', zorder=6)

    # Dummy scatter for colorbar
    sc = ax.scatter([], [], c=[], cmap=cmap, norm=norm)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Range (km)', fontsize=11)
    cbar.set_ticks([0, 50, 100, 150, range_max])

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor='lightgray', edgecolor='gray', label='Charging Station'),
        plt.Line2D([0], [0], color='gray', linewidth=1.5,                 label='Robot moves itself'),
        plt.Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--', label='Transported by human'),
        mpatches.Patch(facecolor='yellow', edgecolor='gray', label='Low range'),
        mpatches.Patch(facecolor='green',  edgecolor='gray', label='High range'),
    ] + [
        mpatches.Patch(facecolor=station_colors[j], edgecolor='gray', label=f'Station {j+1}')
        for j in range(J)
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.9, edgecolor='gray')

    # --- Total robots label (outside plot) ---
    fig.text(0.99, 0.99, f'Total robots: {I}   |   Total stations: {J}',
             fontsize=10, fontweight='bold', color='steelblue',
             ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='steelblue', alpha=0.8))

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")

    plt.show()
    
    
    
def plot_robots_and_clusters(robot_coords, cluster_centers, gamma):
    """
    Plot robot points in blue and cluster centers in red.

    Parameters:
    -----------
    robot_coords : list of tuples or numpy array
        List of (x, y) coordinates for robots
        Example: [(1, 2), (3, 4), (5, 6)] or np.array([[1,2], [3,4], [5,6]])

    cluster_centers : list of tuples or numpy array
        List of (x, y) coordinates for cluster centers
        Example: [(2, 3), (4, 5)] or np.array([[2,3], [4,5]])
    """

    # Convert to numpy arrays if they aren't already
    robot_coords = np.array(robot_coords) if robot_coords is not None and len(robot_coords) > 0 else np.array([])
    cluster_centers = np.array(cluster_centers) if cluster_centers is not None and len(cluster_centers) > 0 else np.array([])

    # Get counts
    n_robots = len(robot_coords) if len(robot_coords) > 0 else 0
    n_centers = len(cluster_centers) if len(cluster_centers) > 0 else 0

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot robots as blue points
    if n_robots > 0:
        # Handle both 1D and 2D arrays
        if robot_coords.ndim == 1:
            robot_coords = robot_coords.reshape(-1, 2)

        robot_x = robot_coords[:, 0]
        robot_y = robot_coords[:, 1]

        # Plot robots as blue points
        ax.scatter(robot_x, robot_y, color='blue', s=100, 
                  label=f'Robots (n={n_robots})', alpha=0.7, 
                  edgecolors='darkblue', linewidth=1.5)

    # Plot cluster centers as red points
    if n_centers > 0:
        # Handle both 1D and 2D arrays
        if cluster_centers.ndim == 1:
            cluster_centers = cluster_centers.reshape(-1, 2)

        center_x = cluster_centers[:, 0]
        center_y = cluster_centers[:, 1]

        # Plot cluster centers as red points (larger and with star marker for emphasis)
        ax.scatter(center_x, center_y, color='red', s=200, 
                  marker='*', label=f'Cluster Centers (k={n_centers})', 
                  edgecolors='darkred', linewidth=2, zorder=5)

    # Add labels and title
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'Robot Positions and Cluster Centers\n({n_robots} robots, {n_centers} clusters, gamma={gamma})', 
                 fontsize=14, fontweight='bold')

    # Add legend with counts
    ax.legend(fontsize=11, loc='best', framealpha=0.9)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set equal aspect ratio to maintain coordinate proportions
    ax.set_aspect('equal', adjustable='box')

    # Add some padding around the points
    all_points = []
    if n_robots > 0:
        all_points.extend(robot_coords)
    if n_centers > 0:
        all_points.extend(cluster_centers)

    if all_points:
        all_points = np.array(all_points)
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

        x_range = x_max - x_min
        y_range = y_max - y_min
        padding_x = max(x_range * 0.1, 1) if x_range > 0 else 5
        padding_y = max(y_range * 0.1, 1) if y_range > 0 else 5
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)

    # Show the plot
    plt.tight_layout()
    plt.show()
    
def plot_robots_and_clusters_assigned(robot_coords, cluster_centers, a_matrix=None, h_matrix=None):
    """
    Plot robot points in blue and cluster centers in red, with assignment lines.

    Parameters:
    -----------
    robot_coords : list of tuples or numpy array
        List of (x, y) coordinates for robots
        Example: [(1, 2), (3, 4), (5, 6)] or np.array([[1,2], [3,4], [5,6]])

    cluster_centers : list of tuples or numpy array
        List of (x, y) coordinates for cluster centers
        Example: [(2, 3), (4, 5)] or np.array([[2,3], [4,5]])

    a_matrix : numpy array, optional
        Binary matrix a[i,j] = 1 if robot i assigned to station j by drone
        Shape: (n_robots, n_clusters)

    h_matrix : numpy array, optional
        Binary matrix h[i,j] = 1 if robot i assigned to station j by human
        Shape: (n_robots, n_clusters)
    """

    # Convert to numpy arrays if they aren't already
    robot_coords = np.array(robot_coords) if robot_coords is not None and len(robot_coords) > 0 else np.array([])
    cluster_centers = np.array(cluster_centers) if cluster_centers is not None and len(cluster_centers) > 0 else np.array([])

    # Get counts
    n_robots = len(robot_coords) if len(robot_coords) > 0 else 0
    n_centers = len(cluster_centers) if len(cluster_centers) > 0 else 0

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw assignment lines first (so they appear behind the points)
    if a_matrix is not None and h_matrix is not None and n_robots > 0 and n_centers > 0:
        # Ensure matrices have correct shape
        if a_matrix.shape == (n_robots, n_centers) and h_matrix.shape == (n_robots, n_centers):

            # Drone assignments (dashed black lines)
            drone_lines = []
            human_lines = []

            for i in range(n_robots):
                robot_x, robot_y = robot_coords[i]

                for j in range(n_centers):
                    center_x, center_y = cluster_centers[j]

                    # Check for drone assignment (a_ij = 1)
                    if a_matrix[i, j] == 1:
                        line = ax.plot([robot_x, center_x], [robot_y, center_y], 
                                     'k--', linewidth=1.5, alpha=0.6, 
                                     label='Drone Assignment' if i == 0 and j == 0 else '')[0]
                        drone_lines.append(line)

                    # Check for human assignment (h_ij = 1)
                    if h_matrix[i, j] == 1:
                        line = ax.plot([robot_x, center_x], [robot_y, center_y], 
                                     'k-', linewidth=2, alpha=0.7,
                                     label='Human Assignment' if i == 0 and j == 0 else '')[0]
                        human_lines.append(line)

    elif a_matrix is not None and n_robots > 0 and n_centers > 0:
        # Only drone assignments provided
        for i in range(n_robots):
            robot_x, robot_y = robot_coords[i]
            for j in range(n_centers):
                if a_matrix[i, j] == 1:
                    center_x, center_y = cluster_centers[j]
                    ax.plot([robot_x, center_x], [robot_y, center_y], 
                          'k--', linewidth=1.5, alpha=0.6,
                          label='Drone Assignment' if i == 0 and j == 0 else '')

    elif h_matrix is not None and n_robots > 0 and n_centers > 0:
        # Only human assignments provided
        for i in range(n_robots):
            robot_x, robot_y = robot_coords[i]
            for j in range(n_centers):
                if h_matrix[i, j] == 1:
                    center_x, center_y = cluster_centers[j]
                    ax.plot([robot_x, center_x], [robot_y, center_y], 
                          'k-', linewidth=2, alpha=0.7,
                          label='Human Assignment' if i == 0 and j == 0 else '')

    # Plot robots as blue points
    if n_robots > 0:
        # Handle both 1D and 2D arrays
        if robot_coords.ndim == 1:
            robot_coords = robot_coords.reshape(-1, 2)

        robot_x = robot_coords[:, 0]
        robot_y = robot_coords[:, 1]

        # Plot robots as blue points
        ax.scatter(robot_x, robot_y, color='blue', s=100, 
                  label=f'Robots (n={n_robots})', alpha=0.8, 
                  edgecolors='darkblue', linewidth=1.5, zorder=3)

    # Plot cluster centers as red points
    if n_centers > 0:
        # Handle both 1D and 2D arrays
        if cluster_centers.ndim == 1:
            cluster_centers = cluster_centers.reshape(-1, 2)

        center_x = cluster_centers[:, 0]
        center_y = cluster_centers[:, 1]

        # Plot cluster centers as red points (larger and with star marker for emphasis)
        ax.scatter(center_x, center_y, color='red', s=200, 
                  marker='*', label=f'Cluster Centers (k={n_centers})', 
                  edgecolors='darkred', linewidth=2, zorder=4)

    # Add labels and title
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)

    # Create title with assignment info
    title = f'Robot Assignments to Cluster Centers\n({n_robots} robots, {n_centers} clusters)'
    if a_matrix is not None and h_matrix is not None:
        n_drone = np.sum(a_matrix)
        n_human = np.sum(h_matrix)
        title += f'\nDrone assignments: {n_drone}, Human assignments: {n_human}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add legend (remove duplicates)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=10, loc='best', framealpha=0.9)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set equal aspect ratio to maintain coordinate proportions
    ax.set_aspect('equal', adjustable='box')

    # Add some padding around the points
    all_points = []
    if n_robots > 0:
        all_points.extend(robot_coords)
    if n_centers > 0:
        all_points.extend(cluster_centers)

    if all_points:
        all_points = np.array(all_points)
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

        x_range = x_max - x_min
        y_range = y_max - y_min
        padding_x = max(x_range * 0.1, 1) if x_range > 0 else 5
        padding_y = max(y_range * 0.1, 1) if y_range > 0 else 5
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)

    # Show the plot
    plt.tight_layout()
    plt.show()