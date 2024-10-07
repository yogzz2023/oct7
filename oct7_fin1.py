import numpy as np
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors  # For interactive data tips
from scipy.stats import chi2
import time

# CVFilter class for Kalman filter
class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))  # Predicted state vector
        self.Pp = np.eye(6)  # Predicted state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.prev_Time = 0
        self.Q = np.eye(6)
        self.Phi = np.eye(6)
        self.Z = np.zeros((3, 1)) 
        self.Z1 = np.zeros((3, 1))  # Measurement vector
        self.Z2 = np.zeros((3, 1)) 
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = 9.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        if not self.first_rep_flag:
            self.Z1 = np.array([[x], [y], [z]])
            self.Sf[0] = x
            self.Sf[1] = y
            self.Sf[2] = z
            self.Meas_Time = time
            self.prev_Time = self.Meas_Time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2 = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time
            dt = self.Meas_Time - self.prev_Time
            self.vx = (self.Z1[0] - self.Z2[0]) / dt
            self.vy = (self.Z1[1] - self.Z2[1]) / dt
            self.vz = (self.Z1[2] - self.Z2[2]) / dt
            self.Meas_Time = time
            self.second_rep_flag = True
        else:
            self.Z = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time

    def predict_step(self, current_time):
        dt = current_time - self.prev_Time
        T_2 = (dt * dt) / 2.0
        T_3 = (dt * dt * dt) / 3.0
        self.Phi[0, 3] = dt
        self.Phi[1, 4] = dt
        self.Phi[2, 5] = dt              
        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        self.Q[3, 0] = T_2
        self.Q[4, 1] = T_2
        self.Q[5, 2] = T_2
        self.Q[3, 3] = dt
        self.Q[4, 4] = dt
        self.Q[5, 5] = dt
        self.Q = self.Q * self.plant_noise
        self.Sp = np.dot(self.Phi, self.Sf)
        self.Pp = np.dot(np.dot(self.Phi, self.Pf), self.Phi.T) + self.Q
        self.Meas_Time = current_time

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sp + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pp)

# Spherical to Cartesian Conversion
def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

# Cartesian to Spherical Conversion
def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x)
    if x > 0.0:
        az = np.pi / 2 - az
    else:
        az = 3 * np.pi / 2 - az

    az = az * 180 / np.pi
    if az < 0.0:
        az = 360 + az
    if az > 360:
        az = az - 360

    return r, az, el

# Reading Measurements from CSV
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            md = float(row[11])
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((mr, ma, me, mt, md, x, y, z))
    return measurements

# Plot Tracks with Sf and Measurements
def plot_tracks(tracks, kalman_filter):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    range_ax = axes[0]
    azimuth_ax = axes[1]
    elevation_ax = axes[2]
    
    # To store the plot data for data tips
    plot_data = []

    # Plot each track's data
    for track in tracks:
        track_id = track['track_id']
        measurements = track['measurements']  # (range, az, el, time, ...)
        Sf = kalman_filter.Sf  # State filter values for the track
        
        times = [m[0][3] for m in measurements]  # Extract time
        ranges = [m[0][0] for m in measurements]  # Extract range
        azimuths = [m[0][1] for m in measurements]  # Extract azimuth
        elevations = [m[0][2] for m in measurements]  # Extract elevation
        
        Sf_ranges = Sf[0, :]  # Kalman filter Sf (range)
        Sf_azimuths = Sf[1, :]  # Kalman filter Sf (azimuth)
        Sf_elevations = Sf[2, :]  # Kalman filter Sf (elevation)

        # Plot range and Sf(range) vs time
        range_ax.plot(times, ranges, label=f'Track {track_id} Range', marker='o', linestyle='-')
        range_ax.plot(times, Sf_ranges, label=f'Track {track_id} Sf(range)', marker='x', linestyle='--')
        
        # Plot azimuth and Sf(azimuth) vs time
        azimuth_ax.plot(times, azimuths, label=f'Track {track_id} Azimuth', marker='o', linestyle='-')
        azimuth_ax.plot(times, Sf_azimuths, label=f'Track {track_id} Sf(azimuth)', marker='x', linestyle='--')
        
        # Plot elevation and Sf(elevation) vs time
        elevation_ax.plot(times, elevations, label=f'Track {track_id} Elevation', marker='o', linestyle='-')
        elevation_ax.plot(times, Sf_elevations, label=f'Track {track_id} Sf(elevation)', marker='x', linestyle='--')

        # Save data for interactive tooltips
        plot_data.append({
            'track_id': track_id,
            'times': times,
            'ranges': ranges,
            'azimuths': azimuths,
            'elevations': elevations
        })

    # Set titles and labels for each axis
    range_ax.set_title('Range and Sf(range) vs Time')
    range_ax.set_xlabel('Time')
    range_ax.set_ylabel('Range')
    range_ax.legend()

    azimuth_ax.set_title('Azimuth and Sf(azimuth) vs Time')
    azimuth_ax.set_xlabel('Time')
    azimuth_ax.set_ylabel('Azimuth')
    azimuth_ax.legend()

    elevation_ax.set_title('Elevation and Sf(elevation) vs Time')
    elevation_ax.set_xlabel('Time')
    elevation_ax.set_ylabel('Elevation')
    elevation_ax.legend()

    # Enable data tips
    enable_data_tips(fig, plot_data)

    plt.tight_layout()
    plt.show()

# Enable Data Tips with mplcursors
def enable_data_tips(fig, plot_data):
    cursor = mplcursors.cursor(hover=True)

    @cursor.connect("add")
    def on_add(sel):
        # Find the nearest point and provide custom data tips
        index = sel.index
        ax = sel.artist.axes
        for data in plot_data:
            if sel.artist in ax.lines:
                track_id = data['track_id']
                time = data['times'][index]
                range_ = data['ranges'][index]
                azimuth = data['azimuths'][index]
                elevation = data['elevations'][index]
                
                sel.annotation.set_text(
                    f'Track ID: {track_id}\n'
                    f'Time: {time}\n'
                    f'Range: {range_}\n'
                    f'Azimuth: {azimuth}\n'
                    f'Elevation: {elevation}'
                )

# Main function to run the process
def main():
    file_path = 'ttk.csv'
    measurements = read_measurements_from_csv(file_path)

    kalman_filter = CVFilter()
    
    # Initialize tracks and populate their measurements (as an example)
    tracks = [{
        'track_id': 1,
        'measurements': measurements
    }]
    
    # Call plotting function after processing
    plot_tracks(tracks, kalman_filter)

if __name__ == "__main__":
    main()
