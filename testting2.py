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
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = chi2.ppf(0.95, 3)  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        if not self.first_rep_flag:
            self.Z1 = np.array([[x], [y], [z]])
            self.Sf[0] = self.Z1[0]
            self.Sf[1] = self.Z1[1]
            self.Sf[2] = self.Z1[2]
            self.Meas_Time = time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2 = np.array([[x], [y], [z]])
            self.Meas_Time = time
            self.second_rep_flag = True
        
    def InitializeMeasurementForFiltering(self, x, y, z, vx, vy, vz, mt):
        self.Z = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q

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

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r=np.sqrt(x**2 + y**2 + z**2)
    el=math.atan(z/np.sqrt(x**2 + y**2))*180/3.14
    az=math.atan(y/x)    

    if x > 0.0:                
        az = 3.14/2 - az
    else:
        az = 3*3.14/2 - az       
        
    az=az*180/3.14 

    if(az<0.0):
        az=(360 + az)
    
    if(az>360):
        az=(az - 360)   
      
    return r,az,el

def cart2sph2(x:float,y:float,z:float,filtered_values_csv):
    for i in range(len(filtered_values_csv)):
    #print(np.array(y))
    #print(np.array(z))
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
      

        # print("Row ",i+1)
        # print("range:", r)
        # print("azimuth", az)
        # print("elevation",el)s
        # print()
  
    return r,az,el
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
            # x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((ma, me, mr, mt))  # Storing (azimuth, elevation, range, time)
    return measurements

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

def chi_square_clustering(group, kalman_filter):
    clusters = []
    for measurement in group:
        Z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
        if kalman_filter.gating(Z):
            clusters.append(measurement)
    return clusters

def generate_hypotheses(clusters, targets):
    hypotheses = []
    for cluster in clusters:
        for target in targets:
            hypotheses.append((cluster, target))
    return hypotheses

def compute_hypothesis_likelihood(hypothesis, filter_instance):
    cluster, target = hypothesis
    Z = np.array([[cluster[0]], [cluster[1]], [cluster[2]]])
    Inn = Z - np.dot(filter_instance.H, target)
    S = np.dot(filter_instance.H, np.dot(filter_instance.Pf, filter_instance.H.T)) + filter_instance.R
    likelihood = np.exp(-0.5 * np.dot(np.dot(Inn.T, np.linalg.inv(S)), Inn))
    return likelihood

def jpda(measurements, targets, kalman_filter):
    valid_clusters = []
    for measurement in measurements:
        Z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
        if kalman_filter.gating(Z):
            valid_clusters.append(measurement)
    
    hypotheses = generate_hypotheses(valid_clusters, targets)
    
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

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'ttk_84.csv'  # Provide the path to your CSV file


csv_file_predicted = "ttk_84.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values
measured_values_csv = df_predicted[['MT', 'MR', 'MA', 'ME']].values

A=cart2sph2(filtered_values_csv[:,1],filtered_values_csv[:,2],filtered_values_csv[:,3],filtered_values_csv)

number= 1000

result=np.divide(A[0],number)

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Form measurement groups based on time
measurement_groups = form_measurement_groups(measurements)

# Lists to store the data for plotting
time_list = []
r_list = []
az_list = []
el_list = []

# Initial targets list
targets = []

# Iterate through measurement groups
for group in measurement_groups:
    # x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
    for i, (ma, me, mr, mt) in enumerate(group):
        x, y, z = mr * np.cos(me * np.pi / 180) * np.sin(ma * np.pi / 180), mr * np.cos(me * np.pi / 180) * np.cos(ma * np.pi / 180), mr * np.sin(me * np.pi / 180)
   
        if not kalman_filter.first_rep_flag:
            kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
        elif kalman_filter.first_rep_flag and not kalman_filter.second_rep_flag:
            if kalman_filter.gating(np.array([[x], [y], [z]])):
                prev_x, prev_y, prev_z = kalman_filter.Z1[0], kalman_filter.Z1[1], kalman_filter.Z1[2]
                dt = mt - kalman_filter.Meas_Time
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
                vz = (z - prev_z) / dt
                kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, mt)
                kalman_filter.second_rep_flag = True
            else:
                kalman_filter.predict_step(mt)
                targets = kalman_filter.Sp.tolist()
                clusters = chi_square_clustering(group, kalman_filter)
                if clusters:
                    best_hypothesis = jpda(clusters, targets, kalman_filter)
                    Z = np.array([[best_hypothesis[0][0]], [best_hypothesis[0][1]], [best_hypothesis[0][2]]])
                    kalman_filter.update_step(Z)

        # Append data for plotting
        time_list.append(mt)
        r_list.append(mr)
        az_list.append(ma)
        # el_list.append(me)
        # r_list.append(r)
        # az_list.append(az)
        # el_list.append(el)
        time_list.append(mt)

# Plot range (r) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor ="white")
#plt.plot(time_list, r_list, color='green', linewidth=2)
plt.scatter(time_list, r_list, label='filtered range (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], result, label='filtered range (track id 31)', color='red', marker='*')
# plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 1],label='measured range (code)', color='blue', marker='o', linestyle='--')
#plt.scatter(closest_measurement[3], closest_measurement[0], label='associated range', color='blue', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Range (r)', color='black')
plt.title('Range vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)

plt.show()

# Plot azimuth (az) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor ="white")
#plt.plot(time_list, az_list, color='green', linewidth=2)
plt.scatter(time_list, az_list, label='filtered azimuth (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], A[1], label='filtered azimuth (track id 31)', color='red', marker='*')
# plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 2],label='measured range (code)', color='blue', marker='o', linestyle='--')
#plt.scatter(closest_measurement[3], closest_measurement[1], label='associated azimuth', color='blue', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Azimuth (az)', color='black')
plt.title('Azimuth vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()

# Plot elevation (el) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor ="white")
#plt.plot(time_list, el_list, color='green', linewidth=2)
plt.scatter(time_list, el_list, label='filtered elevation (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], A[2], label='filtered elevation (track id 31)', color='red', marker='*')
# plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 3],label='measured range (code)', color='blue', marker='o', linestyle='--')
#plt.scatter(closest_measurement[3], closest_measurement[2], label='associated elevation', color='blue', marker='x')
plt.xlabel('Time', color='black')
plt.ylabel('Elevation (el)', color='black')
plt.title('Elevation vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()
