import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2
import mplcursors

# Global variables and constants
state_dim = 3  # Dimension of state vector (x, y, z)

cov_matrix = np.eye(state_dim)
cov_inv = np.linalg.inv(cov_matrix)

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

# Function to convert Cartesian coordinates to spherical coordinates
def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan(z / np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x) * 180 / np.pi
    if az < 0:
        az += 360
    return r, az, el

# Kalman Filter Class
class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

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

    def update_step(self, z, association_weight):
        # Update step
        Inn = z - np.dot(self.H, self.Sf)  # Calculate innovation directly
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)

# Function to perform Chi-square test for track assignment
def chi_square_test(track, report, H, cov_inv):
    cov_matrix = np.eye(state_dim)
    cov_inv = np.linalg.inv(cov_matrix)
    predicted_measurement = np.dot(track, H)
    delta = np.array(report) - predicted_measurement.reshape(-1)  # Calculate innovation directly
    distance_squared = np.dot(np.dot(delta.T, cov_inv), delta)
    return distance_squared < chi2_threshold



# Function to group measurements based on time intervals less than 50 milliseconds
def group_measurements_by_time(measurements, time_threshold):
    measurement_groups = []
    current_group = []

    for i, (r, az, el, mt) in enumerate(measurements):
        if i == 0:
            current_group.append((r, az, el, mt))
            continue
        
        prev_mt = measurements[i - 1][3]
        if mt - prev_mt <= time_threshold:
            current_group.append((r, az, el, mt))
        else:
            measurement_groups.append(current_group)
            current_group = [(r, az, el, mt)]

    # Add the last group
    if current_group:
        measurement_groups.append(current_group)

    return measurement_groups

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            measurements.append((r, az, el, mt))
    return measurements

# Function to initialize tracks from measurement groups
def initialize_tracks(measurement_groups):
    tracks = []
    track_id = 0

    for group in measurement_groups:
        initial_measurement = group[0]
        tracks.append({
            'track_id': track_id,
            'measurements': [initial_measurement],
            'associated_reports': [0]  # Indices of associated reports in the group
        })
        track_id += 1

    return tracks

# Function to update tracks with subsequent measurements using Chi-square test
def update_tracks_with_measurements(tracks, measurement_groups, cov_inv):
    for group in measurement_groups:
        for report_idx, report in enumerate(group):
            assigned = False
            for track in tracks:
                last_measurement = track['measurements'][-1]
                if chi_square_test(last_measurement, report, cov_inv):
                    track['measurements'].append(report)
                    track['associated_reports'].append(report_idx)
                    assigned = True
                    break
            if not assigned:
                tracks.append({
                    'track_id': len(tracks),  # Assign new track ID
                    'measurements': [report],
                    'associated_reports': [report_idx]
                })

def mahalanobis_distance(x, y, cov_inv):
    delta = y - x
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

# Function to form clusters from tracks
def form_clusters(tracks):
    clusters = []
    for track in tracks:
        cluster_tracks = [track['track_id']]
        cluster_reports = track['associated_reports']
        clusters.append((cluster_tracks, cluster_reports))
    return clusters

# Function to generate hypotheses for clusters
def generate_hypotheses(tracks, reports):
    hypotheses = []
    for track in tracks:
        track_id = track['track_id']
        report_indices = track['associated_reports']
        base = len(reports) + 1
        for count in range(base**len(track['measurements'])):
            hypothesis = []
            for i in range(len(track['measurements'])):
                report_idx = (count // (base**i)) % base
                hypothesis.append((track_id, report_idx - 1))
            if is_valid_hypothesis(hypothesis):
                hypotheses.append(hypothesis)
    return hypotheses

# Function to check if a hypothesis is valid
def is_valid_hypothesis(hypothesis):
    non_zero_hypothesis = [val for _, val in hypothesis if val != -1]
    return len(non_zero_hypothesis) == len(set(non_zero_hypothesis)) and len(non_zero_hypothesis) > 0

# Function to calculate probabilities for each hypothesis
def calculate_probabilities(hypotheses, tracks, reports, cov_inv):
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_id, report_idx in hypothesis:
            if report_idx != -1:
                distance = mahalanobis_distance(tracks[track_id]['measurements'][report_idx], reports[report_idx], cov_inv)
                prob *= np.exp(-0.5 * distance**2)
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return probabilities

# Function to get association weights for hypotheses
def get_association_weights(hypotheses, probabilities):
    num_tracks = len(hypotheses[0])
    association_weights = [[] for _ in range(num_tracks)]

    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                association_weights[track_idx].append((report_idx, prob))

    for track_weights in association_weights:
        track_weights.sort(key=lambda x: x[0])  # Sort by report index
        report_probs = {}
        for report_idx, prob in track_weights:
            if report_idx not in report_probs:
                report_probs[report_idx] = prob
            else:
                report_probs[report_idx] += prob
        track_weights[:] = [(report_idx, prob) for report_idx, prob in report_probs.items()]

    return association_weights

# Function to find the most likely association for each report
def find_max_associations(hypotheses, probabilities):
    max_associations = [-1] * len(reports)
    max_probs = [0.0] * len(reports)
    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if report_idx != -1 and prob > max_probs[report_idx]:
                max_probs[report_idx] = prob
                max_associations[report_idx] = track_idx
    return max_associations

# Function to update Kalman filter with the most probable measurement for each track
def update_tracks_with_kalman(tracks, max_associations, reports, kalman_filter):
    for report_idx, track_idx in enumerate(max_associations):
        if track_idx != -1:
            report = reports[report_idx]
            kalman_filter.update_step(report, track_idx)

# Function to convert Cartesian coordinates to spherical coordinates
def cart2sph2(x:float,y:float,z:float,filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i]/np.sqrt(x[i]**2 + y[i]**2))*180/3.14)
        az.append(math.atan(y[i]/x[i]))
         
        if x[i] > 0.0:                
            az[i] = 3.14/2 - az[i]
        else:
            az[i] = 3*3.14/2 - az[i]       
        
        az[i]=az[i]*180/3.14 

        if(az[i]<0.0):
            az[i]=(360 + az[i])
    
        if(az[i]>360):
            az[i]=(az[i] - 360)   

    return r, az, el

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((x, y, z, mt))
    return measurements

# Main script
if __name__ == "__main__":
    # Parameters
    time_threshold = 0.05  # 50 milliseconds threshold
    chi2_threshold = chi2.ppf(0.95, df=state_dim)  # Chi-squared gating threshold for 95% confidence interval

    # Read measurements from CSV file
    csv_file_path = 'ttk_84_test.csv'  # Replace with your CSV file path
    measurements = read_measurements_from_csv(csv_file_path)

    # Group measurements based on time intervals less than 50 milliseconds
    measurement_groups = group_measurements_by_time(measurements, time_threshold)

    # Initialize Kalman Filter
    kalman_filter = CVFilter()



    # Initialize tracks with measurement groups
    tracks = initialize_tracks(measurement_groups)

    # Update tracks with subsequent measurements using Chi-square test
    for group in measurement_groups:
        update_tracks_with_measurements(tracks, group, kalman_filter.H)

    # Form clusters from tracks
    clusters = form_clusters(tracks)

    # Generate hypotheses for clusters
    hypotheses = generate_hypotheses(tracks, measurements)

    # Calculate probabilities for hypotheses
    probabilities = calculate_probabilities(hypotheses, tracks, measurements, np.linalg.inv(kalman_filter.R))

    # Get association weights for hypotheses
    association_weights = get_association_weights(hypotheses, probabilities)

    # Find the most likely association for each report
    max_associations = find_max_associations(hypotheses, probabilities)

    # Update Kalman filter with the most probable measurement for each track
    update_tracks_with_kalman(tracks, max_associations, measurements, kalman_filter)

    # Plotting code (example)
    # Assuming you have functions to convert filtered_values_csv to r, az, el
    # Replace with your plotting logic
    filtered_values_csv = pd.read_csv('ttk_84_test.csv')  # Replace with your filtered data CSV path
    r, az, el = cart2sph2(filtered_values_csv['F_X'].values, filtered_values_csv['F_Y'].values, filtered_values_csv['F_Z'].values, filtered_values_csv)

    # Plot range (r) vs. time
    plt.figure(figsize=(12, 6))
    plt.scatter(filtered_values_csv['F_TIM'] + 0.013, r, label='Filtered Range (Track ID 31)', color='red', marker='*')
    # Add other plots as needed
    plt.xlabel('Time')
    plt.ylabel('Range (r)')
    plt.title('Range vs. Time')
    plt.grid(True)
    plt.legend()
    mplcursors.cursor(hover=True)
    plt.tight_layout()
    plt.show()

    # Similar plotting for azimuth (az) and elevation (el)

