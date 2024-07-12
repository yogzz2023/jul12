import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2
import mplcursors

# Define the measurement noise parameters
sig_r = 30  # Range measurement noise standard deviation
sig_a = 5e-3  # Azimuth measurement noise standard deviation (in radians)
sig_e = 5e-3  # Elevation measurement noise standard deviation (in radians)

# Kalman Filter Class
class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.diag([sig_r**2, sig_a**2, sig_e**2])  # Measurement noise covariance

    def initialize_filter_state(self, r, az, el, vx, vy, vz, time):
        # Initialize filter state
        x, y, z = self.sph2cart(az, el, r)
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

    def update_step(self, measurement):
        # Update step with JPDA
        z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
        Inn = z - np.dot(self.H, self.Sf)  # Calculate innovation directly
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)
        print("Updated filter state:")
        print("Sf:", self.Sf)
        print("Pf:", self.Pf)

    def sph2cart(self, az, el, r):
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
            mr = float(row[7])  # MR column (range)
            ma = float(row[8])  # MA column (azimuth in radians)
            me = float(row[9])  # ME column (elevation in radians)
            mt = float(row[10])  # MT column (time)
            measurements.append((mr, ma, me, mt))
    return measurements

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    x = r * np.cos(el) * np.sin(az)
    y = r * np.cos(el) * np.cos(az)
    z = r * np.sin(el)
    return x, y, z

# Function to form measurement groups based on time intervals less than 50 milliseconds
def form_measurement_groups(measurements):
    groups = []
    current_group = []
    prev_time = measurements[0][3]  # Initial time of the first measurement
    for i in range(len(measurements)):
        mt = measurements[i][3]
        if mt - prev_time < 5.4961000000003:
            current_group.append(measurements[i])
        else:
            groups.append(current_group)
            current_group = [measurements[i]]
        prev_time = mt
    groups.append(current_group)  # Add the last group
    print("grppppp",groups)
    return groups

# Function to form clusters from measurement groups
def form_clusters(measurement_groups):
    clusters = []
    for group in measurement_groups:
        cluster = [measurement[:3] for measurement in group]  # Extract (range, azimuth, elevation)
        clusters.append(cluster)
    return clusters

# Function to generate hypotheses for each cluster
def generate_hypotheses(clusters):
    hypotheses = []
    for cluster in clusters:
        cluster_hypotheses = []
        for measurement in cluster:
            cluster_hypotheses.append((measurement[0], measurement[1], measurement[2]))  # (range, azimuth, elevation)
        hypotheses.append(cluster_hypotheses)
    return hypotheses

# Function to calculate joint probabilities for each hypothesis
def calculate_joint_probabilities(clusters, hypotheses, kalman_filter):
    joint_probabilities = []
    for cluster, cluster_hypotheses in zip(clusters, hypotheses):
        cluster_joint_probabilities = []
        for measurement in cluster:
            max_prob = 0.0
            for hypothesis in cluster_hypotheses:
                x, y, z = kalman_filter.sph2cart(hypothesis[1], hypothesis[2], hypothesis[0])
                distance = np.linalg.norm(np.array([x, y, z]) - np.array(measurement))
                joint_prob = np.exp(-0.5 * (distance**2))
                if joint_prob > max_prob:
                    max_prob = joint_prob
            cluster_joint_probabilities.append(max_prob)
        joint_probabilities.append(cluster_joint_probabilities)
    return joint_probabilities

# Function to find the most probable association for each measurement
def find_max_probable_associations(joint_probabilities):
    max_associations = []
    for cluster_probs in joint_probabilities:
        max_index = np.argmax(cluster_probs)
        max_associations.append(max_index)
    return max_associations

# Define the path to your CSV file containing measurements
csv_file_path = 'ttk_84.csv'  # Replace with your file path

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Form measurement groups based on time intervals less than 50 milliseconds
measurement_groups = form_measurement_groups(measurements)

# Form clusters from measurement groups
clusters = form_clusters(measurement_groups)
print("clust",clusters)

# Generate hypotheses for each cluster
hypotheses = generate_hypotheses(clusters)

print("hypotheses",hypotheses)


# Calculate joint probabilities for each hypothesis
joint_probabilities = calculate_joint_probabilities(clusters, hypotheses, kalman_filter)

print("joint_probabilities",joint_probabilities)


# Find the most probable association for each measurement
max_associations = find_max_probable_associations(joint_probabilities)

print("max_associations",max_associations)


# Print or use max_associations as needed for further processing
print("Max Probable Associations:", max_associations)
