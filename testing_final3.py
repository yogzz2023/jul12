import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from scipy.stats import chi2

# Define lists to store results
r = []
el = []
az = []

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
        self.Z = np.zeros((3, 1))  # Measurement vector
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = 9.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        if not self.first_rep_flag:
            self.Sf[0] = x
            self.Sf[1] = y
            self.Sf[2] = z
            self.Meas_Time = time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Sf[3] = vx
            self.Sf[4] = vy
            self.Sf[5] = vz
            self.Meas_Time = time
            self.second_rep_flag = True

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q
        self.Meas_Time = current_time

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)

    def gating(self, Z):
        Inn = Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        d2 = np.dot(np.dot(np.transpose(Inn), np.linalg.inv(S)), Inn)
        return d2 < self.gate_threshold


def form_measurement_groups(measurements, max_time_diff=50):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]

    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]

    if current_group:
        measurement_groups.append(current_group)

    return measurement_groups


def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            x = float(row[0])  # Example: X coordinate column
            y = float(row[1])  # Example: Y coordinate column
            z = float(row[2])  # Example: Z coordinate column
            mt = float(row[3])  # Example: Time column
            measurements.append((x, y, z, mt))
    return measurements


def chi_square_clustering(group, kalman_filter):
    clusters = []
    for measurement in group:
        Z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
        if kalman_filter.gating(Z):
            clusters.append(measurement)
    return clusters


def form_clusters(measurements, kalman_filter):
    valid_clusters = []
    for measurement in measurements:
        Z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
        if kalman_filter.gating(Z):
            valid_clusters.append(measurement)
    return valid_clusters


def generate_hypotheses(clusters):
    hypotheses = []
    for cluster in clusters:
        hypotheses.append(cluster)
    return hypotheses


def compute_hypothesis_likelihood(hypothesis, kalman_filter):
    Z = np.array([[hypothesis[0]], [hypothesis[1]], [hypothesis[2]]])
    Inn = Z - np.dot(kalman_filter.H, kalman_filter.Sf)
    S = np.dot(kalman_filter.H, np.dot(kalman_filter.Pf, kalman_filter.H.T)) + kalman_filter.R
    likelihood = np.exp(-0.5 * np.dot(np.dot(Inn.T, np.linalg.inv(S)), Inn))
    return likelihood


def jpda(clusters, kalman_filter):
    hypotheses = generate_hypotheses(clusters)

    if not hypotheses:
        return None

    hypothesis_likelihoods = [compute_hypothesis_likelihood(h, kalman_filter) for h in hypotheses]
    total_likelihood = sum(hypothesis_likelihoods)

    if total_likelihood == 0:
        marginal_probabilities = [1.0 / len(hypotheses)] * len(hypotheses)
    else:
        marginal_probabilities = [likelihood / total_likelihood for likelihood in hypothesis_likelihoods]

    best_hypothesis_index = np.argmax(marginal_probabilities)
    best_hypothesis = hypotheses[best_hypothesis_index]

    return best_hypothesis


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


time_list=[]

def main():
    # File path for measurements CSV
    file_path = 'ttk_84_test.csv'

    # Read measurements from CSV
    measurements = read_measurements_from_csv(file_path)

    # Form measurement groups based on time intervals less than 50 milliseconds
    measurement_groups = form_measurement_groups(measurements, max_time_diff=50)

    # Initialize Kalman filter
    kalman_filter = CVFilter()

    # Process each group of measurements
    for group_idx, group in enumerate(measurement_groups):
        print(f"Processing measurement group {group_idx + 1}...")

        for i, (x, y, z, mt) in enumerate(group):
            print(f"Measurement {i + 1}: (x={x}, y={y}, z={z}, t={mt})")

            if not kalman_filter.first_rep_flag:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
                print("Initialized filter state:", kalman_filter.Sf.flatten())
            elif kalman_filter.first_rep_flag and not kalman_filter.second_rep_flag:
                if i > 0:
                    x_prev, y_prev, z_prev = group[i - 1][:3]
                    dt = mt - group[i - 1][3]
                    vx = (x - x_prev) / dt
                    vy = (y - y_prev) / dt
                    vz = (z - z_prev) / dt
                    kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, mt)
                    print("Initialized filter state:", kalman_filter.Sf.flatten())
            else:
                kalman_filter.predict_step(mt)
                clusters = chi_square_clustering(group, kalman_filter)
                print(f"Clusters formed: {len(clusters)}")

                if clusters:
                    best_hypothesis = jpda(clusters, kalman_filter)
                    print("best_hypothesis:",best_hypothesis)
                    if best_hypothesis:
                        Z = np.array([[best_hypothesis[0]], [best_hypothesis[1]], [best_hypothesis[2]]])
                        kalman_filter.update_step(Z)
                        print("Updated filter state:", kalman_filter.Sf.flatten())

                        # Convert to spherical coordinates for plotting
                        r_val, az_val, el_val = cart2sph(best_hypothesis[0], best_hypothesis[1], best_hypothesis[2])
                        r.append(r_val)
                        az.append(az_val)
                        el.append(el_val)
                        time_list.append(mt)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.scatter(az, el, r, c=r, cmap='plasma', marker='o')

    # Add labels and title
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Elevation (deg)')
    ax.set_zlabel('Range')
    ax.set_title('Object Tracking in 3D Space')

    # Adding cursor information
    mplcursors.cursor(hover=True)

    plt.show()

if __name__ == "__main__":
    main()
