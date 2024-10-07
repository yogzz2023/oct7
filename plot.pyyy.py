import plotly.graph_objects as go
from plotly.offline import iplot

# ... (rest of your code remains the same)

# Prepare data for plots
track_ids = []
range_values = []
azimuth_values = []
elevation_values = []
time_values = []
sf_range_values = []
sf_azimuth_values = []
sf_elevation_values = []

for track_id, track in enumerate(tracks):
    for measurement, state in track['measurements']:
        range_val, azimuth_val, elevation_val, time_val, doppler_val = measurement
        track_ids.append(track_id)
        range_values.append(range_val)
        azimuth_values.append(azimuth_val)
        elevation_values.append(elevation_val)
        time_values.append(time_val)
        sf_range_values.append(hit_counts.get(track_id, 0))
        sf_azimuth_values.append(hit_counts.get(track_id, 0))
        sf_elevation_values.append(hit_counts.get(track_id, 0))

# Create plots
fig1 = go.Figure(data=[
    go.Scatter(
        x=time_values,
        y=range_values,
        mode='markers',
        hoverinfo='text',
        hovertext=[f'Track ID: {track_id}<br>Range: {range_val}<br>Azimuth: {azimuth_val}<br>Elevation: {elevation_val}<br>Time: {time_val}' for track_id, range_val, azimuth_val, elevation_val, time_val in zip(track_ids, range_values, azimuth_values, elevation_values, time_values)]
    ),
    go.Scatter(
        x=time_values,
        y=sf_range_values,
        mode='lines',
        line=dict(color='red')
    )
])

fig2 = go.Figure(data=[
    go.Scatter(
        x=time_values,
        y=azimuth_values,
        mode='markers',
        hoverinfo='text',
        hovertext=[f'Track ID: {track_id}<br>Range: {range_val}<br>Azimuth: {azimuth_val}<br>Elevation: {elevation_val}<br>Time: {time_val}' for track_id, range_val, azimuth_val, elevation_val, time_val in zip(track_ids, range_values, azimuth_values, elevation_values, time_values)]
    ),
    go.Scatter(
        x=time_values,
        y=sf_azimuth_values,
        mode='lines',
        line=dict(color='red')
    )
])

fig3 = go.Figure(data=[
    go.Scatter(
        x=time_values,
        y=elevation_values,
        mode='markers',
        hoverinfo='text',
        hovertext=[f'Track ID: {track_id}<br>Range: {range_val}<br>Azimuth: {azimuth_val}<br>Elevation: {elevation_val}<br>Time: {time_val}' for track_id, range_val, azimuth_val, elevation_val, time_val in zip(track_ids, range_values, azimuth_values, elevation_values, time_values)]
    ),
    go.Scatter(
        x=time_values,
        y=sf_elevation_values,
        mode='lines',
        line=dict(color='red')
    )
])

# Show plots
iplot(fig1)
iplot(fig2)
iplot(fig3)


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import math
import csv
import pandas as pd
from scipy.stats import chi2
import time
import matplotlib.pyplot as plt
import mplcursors

# ... (keep all the existing code up to the main() function)

def main():
    file_path = 'ttk.csv'
    measurements = read_measurements_from_csv(file_path)

    kalman_filter = CVFilter()
    measurement_groups = form_measurement_groups(measurements, max_time_diff=0.050)

    tracks = []
    track_id_list = []
    filter_states = []

    doppler_threshold = 100
    range_threshold = 100
    firm_threshold = 3
    mode = '3-state'

    firm_threshold = select_initiation_mode(mode)
    # Initialize variables outside the loop
    miss_counts = {}
    hit_counts = {}
    firm_ids = set()
    state_map = {}
    state_transition_times = {}
    progression_states = {
        3: ['Poss1', 'Tentative1', 'Firm'],
        5: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Firm'],
        7: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Tentative3', 'Firm']
    }[firm_threshold]

    last_check_time = 0
    check_interval = 0.0005  # 0.5 ms

    # Create dictionaries to store measurement and filter state data for each track
    track_measurements = {}
    track_filter_states = {}

    for group_idx, group in enumerate(measurement_groups):
        print(f"Processing measurement group {group_idx + 1}...")
        
        current_time = group[0][3]  # Assuming the time is at index 3 of each measurement
        
        # ... (keep the rest of the code in the main loop)

        # After processing each measurement group, store the measurements and filter states
        for track_id, track in enumerate(tracks):
            if track_id not in track_measurements:
                track_measurements[track_id] = []
            if track_id not in track_filter_states:
                track_filter_states[track_id] = []
            
            track_measurements[track_id].append((current_time, *track['measurements'][-1][0][:3]))
            track_filter_states[track_id].append((current_time, *kalman_filter.Sf[:3, 0]))

    # After the main loop, create the plots
    plot_measurements_and_filter_states(track_measurements, track_filter_states)

    # ... (keep the rest of the code for CSV output)

def plot_measurements_and_filter_states(track_measurements, track_filter_states):
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    labels = ['Range', 'Azimuth', 'Elevation']
    
    for track_id in track_measurements.keys():
        measurements = np.array(track_measurements[track_id])
        filter_states = np.array(track_filter_states[track_id])
        
        for i, (ax, label) in enumerate(zip(axs, labels)):
            meas_line = ax.plot(measurements[:, 0], measurements[:, i+1], 'o-', label=f'Track {track_id} Measured {label}')
            filter_line = ax.plot(filter_states[:, 0], filter_states[:, i+1], 's-', label=f'Track {track_id} Filtered {label}')
            
            ax.set_xlabel('Time')
            ax.set_ylabel(label)
            ax.set_title(f'{label} vs Time')
            ax.legend()
    
    plt.tight_layout()
    
    # Add data tips
    cursor = mplcursors.cursor(hover=True)
    
    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        for track_id in track_measurements.keys():
            measurements = np.array(track_measurements[track_id])
            filter_states = np.array(track_filter_states[track_id])
            
            meas_dist = np.linalg.norm(measurements[:, [0, sel.artist.axes.get_ylabel().split()[0].lower() == l for l in ['range', 'azimuth', 'elevation']]] - [x, y], axis=1)
            filter_dist = np.linalg.norm(filter_states[:, [0, sel.artist.axes.get_ylabel().split()[0].lower() == l for l in ['range', 'azimuth', 'elevation']]] - [x, y], axis=1)
            
            if np.min(meas_dist) < 0.1 or np.min(filter_dist) < 0.1:
                if np.min(meas_dist) < np.min(filter_dist):
                    idx = np.argmin(meas_dist)
                    data = measurements[idx]
                    data_type = "Measured"
                else:
                    idx = np.argmin(filter_dist)
                    data = filter_states[idx]
                    data_type = "Filtered"
                
                sel.annotation.set_text(f"Track ID: {track_id}\n"
                                        f"Time: {data[0]:.3f}\n"
                                        f"Range: {data[1]:.3f}\n"
                                        f"Azimuth: {data[2]:.3f}\n"
                                        f"Elevation: {data[3]:.3f}\n"
                                        f"Type: {data_type}")
                return
    
    plt.show()

if __name__ == "__main__":
    main()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    