# Khethiwe Cele

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import linregress, zscore

# Step 1: Load the .dat file
file_path = Path('C:/Users/Student/Downloads/Park.dat')
data = pd.read_csv(file_path, delimiter='\t')
data.columns = data.columns.str.strip()  # Strip any extra spaces from column headers

# Step 2: Inspect and Clean the Data
# Check for missing values
missing_values = data.isnull().sum()
print("Missing values per column:\n", missing_values)

# Remove rows with missing values
data = data.dropna()

# Extract columns
digital_field = data['B_digital_(T)']
analog_field = data['B_analog_(T)']
moment = data['moment_(emu)']
temperature = data['T_VTI_(K)']
amplitude = data['sense_amplitude']
time = data['Time']

# Step 3: Error Detection in Magnetization Data
# Calculate Z-scores to detect outliers
moment_zscores = zscore(moment)
outlier_indices = np.where(np.abs(moment_zscores) > 3)[0]  # Flag data with Z-scores > 3 as outliers

if len(outlier_indices) > 0:
    print(f"Outliers detected in 'moment_(emu)' at indices: {outlier_indices}")
    print("Outlier values:", moment.iloc[outlier_indices])
    # Optionally remove outliers
    moment = moment.drop(outlier_indices)
    temperature = temperature.drop(outlier_indices)

# Step 4: Linear Fit for Clean Data
def linear_fit(x, y):
    """Fit a straight line to the data and return fitted y-values."""
    slope, intercept, _, _, _ = linregress(x, y)
    return slope * x + intercept

# Fit linear relationships
moment_fit = linear_fit(temperature, moment)

# Step 5: Plot Magnetization vs Temperature
plt.figure(figsize=(10, 6))
plt.plot(temperature, moment, 'o', label='Clean Data', alpha=0.6)
plt.plot(temperature, moment_fit, '-', color='red', label='Linear Fit')
plt.xlabel('Temperature (K)')
plt.ylabel('Magnetic Moment (emu)')
plt.title('Magnetic Moment vs. Temperature (Cleaned Data)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Analog vs. Digital Magnetic Field with Error Handling
analog_field = analog_field.replace([np.inf, -np.inf], np.nan).dropna()  # Replace infinite values
digital_field = digital_field[:len(analog_field)]  # Align data lengths

analog_fit = linear_fit(digital_field, analog_field)

plt.figure(figsize=(10, 6))
plt.scatter(digital_field, analog_field, label='Clean Data', alpha=0.6)
plt.plot(digital_field, analog_fit, '-', color='red', label='Linear Fit')
plt.xlabel('Digital Magnetic Field (T)')
plt.ylabel('Analog Magnetic Field (T)')
plt.title('Analog vs. Digital Magnetic Field (Cleaned Data)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Lock-In Amplitude vs Time (Error Checked)
amplitude = amplitude.replace([np.inf, -np.inf], np.nan).dropna()  # Replace infinite values
time = time[:len(amplitude)]  # Align data lengths

amplitude_fit = linear_fit(time, amplitude)

plt.figure(figsize=(10, 6))
plt.plot(time, amplitude, 'o', label='Clean Data', alpha=0.6)
plt.plot(time, amplitude_fit, '-', color='red', label='Linear Fit')
plt.xlabel('Time (s)')
plt.ylabel('Lock-In Amplitude')
plt.title('Lock-In Amplitude vs. Time (Cleaned Data)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
