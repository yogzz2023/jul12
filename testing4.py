import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.stats import chi2
import mplcursors


# Define the measurement noise parameters
sig_r = 30  # Range measurement noise standard deviation
sig_a = 5   # Azimuth measurement noise standard deviation
sig_e_sqr = 5  # Square of the elevation measurement noise standard deviation

# Define the measurement and track parameters
state_dim = 3  # 3D state (e.g., x, y, z)

# Chi-squared gating threshold for 95% confidence interval
chi2_threshold = chi2.ppf(0.95, df=state_dim)

def mahalanobis_distance(x, y, cov_inv):
    delta = y - x
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.diag([sig_r**2, sig_a**2, sig_e_sqr])  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3,1))
        self.tracks = []  # List to store track states

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
            # self.Sf[3] = vx
            # self.Sf[4] = vy
            # self.Sf[5] = vz
            # self.Meas_Time = time
            print("Initialized filter state:")
            print("Sf:", self.Sf)
            print("Pf:", self.Pf)
            self.second_rep_flag = True
        
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
        # Perform Joint Probabilistic Data Association (JPDA)
        association_list = []
        for i, track in enumerate(self.tracks):  # Use self.tracks instead of undefined tracks
            distance = mahalanobis_distance(track, measurement, np.linalg.inv(self.R))
            if distance < chi2_threshold:
                association_list.append((i, distance))
        
        # Calculate association probabilities
        probabilities = np.zeros(len(self.tracks))
        for idx, _ in association_list:
            probabilities[idx] = 1.0 / len(association_list)
        
        # Update using the best track association based on JPDA
        if association_list:
            best_track_idx, _ = min(association_list, key=lambda x: x[1])
            K = np.dot(self.Pf, np.dot(self.H.T, np.linalg.inv(np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R)))
            self.Sf = self.Sf + np.dot(K, (measurement - np.dot(self.H, self.Sf)))
            self.Pf = np.dot((np.eye(6) - np.dot(K, self.H)), self.Pf)
            print(f"Updated filter state with measurement {measurement}:")
            print("Sf:", self.Sf)
            print("Pf:", self.Pf)
        else:
            print("No valid association found.")

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
    r=[]
    az=[]
    el=[]
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
# Function to read measurements from CSV file and form groups based on time intervals
def read_measurements_from_csv(file_path, time_interval=50):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        group = []
        prev_time = None
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            
            if prev_time is None or mt - prev_time <= time_interval:
                group.append((x, y, z, mt))
            else:
                measurements.append(group)
                group = [(x, y, z, mt)]
            prev_time = mt
        
        if group:
            measurements.append(group)
    
    return measurements

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'ttk_84_test.csv'  # Provide the path to your CSV file

csv_file_predicted = "ttk_84_test.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values
measured_values_csv = df_predicted[['MT', 'MR', 'MA', 'ME']].values

A=cart2sph2(filtered_values_csv[:,1],filtered_values_csv[:,2],filtered_values_csv[:,3],filtered_values_csv)

number= 1000

result=np.divide(A[0],number)

# Read measurements from CSV file and form groups based on time intervals less than 50 milliseconds
measurement_groups = read_measurements_from_csv(csv_file_path, time_interval=0.050)

# Lists to store data for plotting
time_list = []
r_list = []
az_list = []
el_list = []

# Iterate through measurement groups
for group_idx, group in enumerate(measurement_groups):
    print(f"Processing Measurement Group {group_idx + 1}")
    
    # Iterate through measurements in the group
    for m_idx, (x, y, z, mt) in enumerate(group):
        if m_idx == 0:
            # Initialize filter state with the first measurement in the group
            kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
        else:
            # Use the second measurement from any group for velocity computation
            prev_x, prev_y, prev_z, prev_mt = group[m_idx - 1]
            dt = mt - prev_mt
            vx = (x - prev_x) / dt
            vy = (y - prev_y) / dt
            vz = (z - prev_z) / dt
            kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, mt)

        # Predict step for the current measurement
        kalman_filter.predict_step(mt)

        # Perform JPDA and update step with the measurement
        kalman_filter.update_step(np.array([x, y, z]))

        # Append data for plotting
        time_list.append(mt)
        r_list.append(np.sqrt(x**2 + y**2 + z**2))  # Compute range (r)
        az_list.append(math.atan2(y, x) * 180 / np.pi)  # Compute azimuth (az)
        el_list.append(math.atan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi)  # Compute elevation (el)

# Plot range (r) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor ="white")
#plt.plot(time_list, r_list, color='green', linewidth=2)
plt.scatter(time_list, r_list, label='filtered range (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], result, label='filtered range (track id 31)', color='red', marker='*')
plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 1],label='measured range (code)', color='blue', marker='o', linestyle='--')
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
plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 2],label='measured range (code)', color='blue', marker='o', linestyle='--')
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
plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 3],label='measured range (code)', color='blue', marker='o', linestyle='--')
#plt.scatter(closest_measurement[3], closest_measurement[2], label='associated elevation', color='blue', marker='x')
plt.xlabel('Time', color='black')
plt.ylabel('Elevation (el)', color='black')
plt.title('Elevation vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()