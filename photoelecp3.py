import matplotlib.pyplot as plt
import pandas as pd

file_path = 'scope_0.csv'
data = pd.read_csv(file_path)

data.columns = ['Time', 'Channel_1_Voltage', 'Channel_2_Voltage']
data = data.drop(0)  # Drop the first row which contains unit descriptions

data['Time'] = pd.to_numeric(data['Time'], errors='coerce')
data['Channel_1_Voltage'] = pd.to_numeric(data['Channel_1_Voltage'], errors='coerce')
data['Channel_2_Voltage'] = pd.to_numeric(data['Channel_2_Voltage'], errors='coerce')

# Drop rows with NaN values
data = data.dropna().reset_index(drop=True)

voltage_jump_threshold = 50 
time_after = -690e-6  

significant_jump_index = data[(data['Time'] > time_after) & (data['Channel_2_Voltage'].diff().abs() > voltage_jump_threshold)].index[0]
time_significant_jump = data['Time'].iloc[significant_jump_index]

print(f"Time of high signal jump: {time_significant_jump} s")


# find when the voltage of the first channel stabilizes to a certain threhold

voltage_stabilization_threshold = 0.01
time_after = -665e-6   # updated time to start looking after
window_size = 8

# Calculate the rolling difference to smooth out fluctuations
data['Voltage_Diff'] = data['Channel_1_Voltage'].diff().abs()

# Find indices where the rolling window meets the threshold condition
stabilization_indices = data[(data['Time'] > time_after) &
                             (data['Voltage_Diff'].rolling(window=window_size).mean() < voltage_stabilization_threshold)].index

stabilization_index = stabilization_indices[0]
time_stabilization = data['Time'].iloc[stabilization_index]
print(f"Time of stabilization: {time_stabilization} s")

# find the time difference between the significant jump and stabilization
time_difference = time_stabilization - time_significant_jump

print(f"Time difference between high signal and stabilization: {time_difference} s")

#### find P_e

#given

P_LED = 60e-3
A_PC = 3.23e-4
electron_space = 0.3e-9

# find A_e
A_e  = electron_space ** 2
P_e = (P_LED * A_e) / A_PC

## energy absorbed by the photoelectron per second
print(f"Energy absorbed by the photoelectron per second: {P_e} J")


#### find escape time
work_function = 1.5981e-19
escape_time = work_function / P_e

print(f"Escape time: {escape_time} s")


plt.figure(figsize=(12, 6))
plt.plot(data['Time'], data['Channel_1_Voltage'], label='Channel 1 (Phototube)')
plt.plot(data['Time'], data['Channel_2_Voltage'], label='Channel 2 (Oscillator-driven LED)', alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Voltage Transients for Phototube and Oscillator-driven LED")
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
## legend location lower right
plt.legend(loc='upper right')
plt.grid(True)
plt.show()