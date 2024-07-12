import numpy as np
from scipy.stats import chi2
import csv
import math
import matplotlib.pyplot as plt
import pandas as pd

# Define measurement noise parameters
sig_r = 30  # Range measurement noise standard deviation
sig_a = 5 / 1000  # Azimuth measurement noise standard deviation (converted to radians)
sig_e_sqr = (5 / 1000)**2  # Square of elevation measurement noise standard deviation (converted to radians)

# Define the measurement and track parameters
state_dim = 3  # 3D state (e.g., x, y, z)

# Chi-squared gating threshold for 95% confidence interval
chi2_threshold = chi2.ppf(0.95, df=state_dim)

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    x = r * np.cos(el) * np.sin(az)
    y = r * np.cos(el) * np.cos(az)
    z = r * np.sin(el)
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

# Kalman Filter Class
class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def InitializeMeasurementForFiltering(self, x, y, z, vx, vy, vz, mt):
        self.Z = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * 20  # Plant noise covariance
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q

    def update_step(self):
        # Update step with JPDA
        Inn = self.Z - np.dot(self.H, self.Sf)  # Calculate innovation directly
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)

# Function to calculate Mahalanobis distance
def mahalanobis_distance(x, y, cov_inv):
    delta = y - x
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

# Function to generate hypotheses for each cluster
def generate_hypotheses(tracks, reports):
    num_tracks = len(tracks)
    num_reports = len(reports)
    base = num_reports + 1
    
    hypotheses = []
    for count in range(base**num_tracks):
        hypothesis = []
        for track_idx in range(num_tracks):
            report_idx = (count // (base**track_idx)) % base
            hypothesis.append((track_idx, report_idx - 1))
        
        # Check if the hypothesis is valid (each report and track is associated with at most one entity)
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
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                distance = mahalanobis_distance(tracks[track_idx], reports[report_idx], cov_inv)
                prob *= np.exp(-0.5 * distance**2)
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()  # Normalize probabilities
    return probabilities

# Function to get association weights
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
    return max_associations, max_probs

# Process and update filter based on clusters
def process_clusters(tracks, reports, cov_inv, filter_instance):
    association_list = []
    for i, track in enumerate(tracks):
        for j, report in enumerate(reports):
            distance = mahalanobis_distance(track, report, cov_inv)  # Step 1: Chi-squared value check for distance
            if distance < chi2_threshold:
                association_list.append((i, j))  # Step 2: Association List
    
    print("Associations List:", association_list)

    # Clustering reports and tracks based on associations
    clusters = []
    while association_list:  # Step 3: Cluster Formation
        cluster_tracks = set()
        cluster_reports = set()
        stack = [association_list.pop(0)]
        while stack:
            track_idx, report_idx = stack.pop()
            cluster_tracks.add(track_idx)
            cluster_reports.add(report_idx)
            new_assoc = [(t, r) for t, r in association_list if t == track_idx or r == report_idx]
            for assoc in new_assoc:
                if assoc not in stack:
                    stack.append(assoc)
            association_list = [assoc for assoc in association_list if assoc not in new_assoc]
        clusters.append((list(cluster_tracks), list(cluster_reports)))

    print("Clusters:", clusters)

    # Process each cluster and generate hypotheses
    for track_idxs, report_idxs in clusters:
        print("Cluster Tracks:", track_idxs)
        print("Cluster Reports:", report_idxs)
        
        cluster_tracks = [tracks[i] for i in track_idxs]
        cluster_reports = [reports[i] for i in report_idxs]
        hypotheses = generate_hypotheses(cluster_tracks, cluster_reports)
        probabilities = calculate_probabilities(hypotheses, cluster_tracks, cluster_reports, cov_inv)
        association_weights = get_association_weights(hypotheses, probabilities)
        max_associations, max_probs = find_max_associations(hypotheses, probabilities)
        
        print("Hypotheses:")
        print("Tracks/Reports:", ["t" + str(i+1) for i in track_idxs])
        for hypothesis, prob  in zip(hypotheses, probabilities):
            formatted_hypothesis = ["r" + str(report_idxs[r]+1) if r != -1 else "0" for _, r in hypothesis]
            print(f"Hypothesis: {formatted_hypothesis}, Probability: {prob:.4f}")
        
        for track_idx, weights in enumerate(association_weights):
            for report_idx, weight in weights:
                print(f"Track t{track_idxs[track_idx]+1}, Report r{report_idxs[report_idx]+1}: {weight:.4f}")
        
        for report_idx, association in enumerate(max_associations):
            if association != -1:
                print(f"Most likely association for Report r{report_idxs[report_idx]+1}: Track t{track_idxs[association]+1}, Probability: {max_probs[report_idx]:.4f}")
                # Perform the update step for the filter
                meas_x, meas_y, meas_z = reports[report_idxs[report_idx]]
                filter_instance.InitializeMeasurementForFiltering(meas_x, meas_y, meas_z, 0, 0, 0, filter_instance.Meas_Time)
                filter_instance.update_step()

# Initialize the Kalman filter
kf = CVFilter()

# Read measurements from CSV file
measurements = read_measurements_from_csv('ttk_84.csv')

# Initialize the filter with the first measurement
initial_measurement = measurements[0]
initial_x, initial_y, initial_z, initial_time = initial_measurement
kf.initialize_filter_state(initial_x, initial_y, initial_z, 0, 0, 0, initial_time)

# Separate the measurements into tracks and reports
tracks = [np.array([initial_x, initial_y, initial_z])]
reports = [np.array([meas[0], meas[1], meas[2]]) for meas in measurements[1:]]

# Covariance matrix of the measurement errors (identity matrix for simplicity)
cov_matrix = np.eye(state_dim)
cov_inv = np.linalg.inv(cov_matrix)

# Process and update filter based on clusters
process_clusters(tracks, reports, cov_inv, kf)
