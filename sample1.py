import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors

# Define CVFilter class for Kalman Filter implementation
class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3,1))  # Measurement vector

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time
        print("Initialized filter state:")
        print("Sf:", self.Sf)
        print("Pf:", self.Pf)

    def predict_step(self, current_time):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sf = np.dot(Phi, self.Sf)
        self.Pf = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q
        print("Predicted filter state:")
        print("Sf:", self.Sf)
        print("Pf:", self.Pf)

    def update_step(self):
        # Update step with JPDA
        Inn = self.Z - np.dot(self.H, self.Sf)  # Calculate innovation directly
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)
        print("Updated filter state:")
        print("Sf:", self.Sf)
        print("Pf:", self.Pf)

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((x, y, z, mt))
    return measurements

# Function to perform chi-square test
def chi_square_test(track, measurement, R):
    innovation = measurement - track.Sf[:3]
    S = np.dot(track.H, np.dot(track.Pf, track.H.T)) + R
    chi_square = np.dot(innovation.T, np.dot(np.linalg.inv(S), innovation))
    return chi_square

# Function to group measurements based on time
def group_measurements_by_time(measurements, threshold=50):
    groups = []
    current_group = [measurements[0]]
    for i in range(1, len(measurements)):
        if measurements[i][3] - current_group[-1][3] <= threshold:
            current_group.append(measurements[i])
        else:
            groups.append(current_group)
            current_group = [measurements[i]]
    groups.append(current_group)  # Add the last group
    return groups

# Function to cluster tracks based on state vectors (position and velocity)
def cluster_tracks(tracks, threshold=3.0):
    clusters = []
    for track in tracks:
        if len(clusters) == 0:
            clusters.append([track])
        else:
            assigned = False
            for cluster in clusters:
                # Using Euclidean distance between the state vectors as a metric for clustering
                distance = np.linalg.norm(track.Sf[:3] - cluster[0].Sf[:3])
                if distance < threshold:
                    cluster.append(track)
                    assigned = True
                    break
            if not assigned:
                clusters.append([track])
    return clusters

# Function to generate hypotheses for each cluster of tracks
def generate_hypotheses(clusters):
    hypotheses = []
    for cluster in clusters:
        # Example: Hypothesizing that the cluster's tracks will follow their current trajectory
        # Additional hypotheses can be generated based on the application's specific requirements
        hypotheses.append({
            'tracks': cluster,
            'description': f"Cluster of {len(cluster)} tracks"
        })
    return hypotheses

# Function to compute joint probabilities for hypotheses
def compute_joint_probabilities(hypotheses):
    # For simplicity, assuming equal probability for each hypothesis
    num_hypotheses = len(hypotheses)
    joint_probabilities = np.ones(num_hypotheses) / num_hypotheses
    return joint_probabilities

# Main script
if __name__ == "__main__":
    # Initialize CVFilter class and read measurements from CSV
    kalman_filter = CVFilter()
    csv_file_path = 'ttk_84.csv'
    measurements = read_measurements_from_csv(csv_file_path)

    # Group measurements by time
    measurement_groups = group_measurements_by_time(measurements)

    # Initialize tracks with the first group's measurements and process subsequent groups
    tracks = []

    for i, group in enumerate(measurement_groups):
        if i == 0:
            for measurement in group:
                x, y, z, mt = measurement
                vx, vy, vz = 0, 0, 0
                kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, mt)
                tracks.append(kalman_filter)
        else:
            for measurement in group:
                x, y, z, mt = measurement
                assigned = False
                for track in tracks:
                    track.predict_step(mt)
                    Z = np.array([[x], [y], [z]])
                    chi_square_value = chi_square_test(track, Z, track.R)
                    chi_square_threshold = 9.488  # Chi-square threshold for 3 degrees of freedom (p = 0.05)
                    if chi_square_value < chi_square_threshold:
                        track.Z = Z
                        track.update_step()
                        assigned = True
                        break
                if not assigned:
                    vx, vy, vz = 0, 0, 0
                    new_track = CVFilter()
                    new_track.initialize_filter_state(x, y, z, vx, vy, vz, mt)
                    tracks.append(new_track)

    # Cluster tracks based on position and velocity
    clusters = cluster_tracks(tracks)

    # Generate hypotheses for each cluster
    hypotheses = generate_hypotheses(clusters)

    # Compute joint probabilities for hypotheses
    joint_probabilities = compute_joint_probabilities(hypotheses)

    # Plotting the results (example plots)

    # Continue with plotting code as before...

    # Example: Plotting range (r) vs. time
    time_list = []
    r_list = []

    for track in tracks:
        for i in range(len(track.Sf[0])):
            time_list.append(track.Meas_Time)
            r_list.append(track.Sf[0][i])

    plt.figure(figsize=(12, 6))
    plt.scatter(time_list, r_list, label='Filtered Range', color='green', marker='*')
    plt.xlabel('Time')
    plt.ylabel('Range (r)')
    plt.title('Range vs. Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

    # Example: Plotting azimuth (az) vs. time
    az_list = []

    for track in tracks:
        for i in range(len(track.Sf[0])):
            az_list.append(track.Sf[1][i])

    plt.figure(figsize=(12, 6))
    plt.scatter(time_list, az_list, label='Filtered Azimuth', color='blue', marker='*')
    plt.xlabel('Time')
    plt.ylabel('Azimuth (az)')
    plt.title('Azimuth vs. Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

    # Example: Plotting elevation (el) vs. time
    el_list = []

    for track in tracks:
        for i in range(len(track.Sf[0])):
            el_list.append(track.Sf[2][i])

    plt.figure(figsize=(12, 6))
    plt.scatter(time_list, el_list, label='Filtered Elevation', color='red', marker='*')
    plt.xlabel('Time')
    plt.ylabel('Elevation (el)')
    plt.title('Elevation vs. Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()
