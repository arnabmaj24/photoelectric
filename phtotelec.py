import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

#functions:
def sigma_b(x, sigma_y):
    """
    For errors in linear regression

    Calculate sigma_b^2 given x values and their uncertainties sigma_y.
    
    Parameters:
    x (array-like): The x values of the dataset.
    sigma_y (array-like): The uncertainties in the y values associated with each x value.
    
    Returns:
    float: The calculated value of sigma_b^2.
    """
    # Convert inputs to numpy arrays
    x = np.array(x)
    sigma_y = np.array(sigma_y)
    
    # Calculate 1 / sigma_y^2 for each element
    inv_sigma_y_squared = 1 / sigma_y**2
    
    # Calculate the terms needed for Delta
    sum_inv_sigma_y_squared = np.sum(inv_sigma_y_squared)
    sum_x_squared_over_sigma_y_squared = np.sum((x**2) * inv_sigma_y_squared)
    sum_x_over_sigma_y_squared = np.sum(x * inv_sigma_y_squared)
    
    # Calculate Delta
    delta = sum_inv_sigma_y_squared * sum_x_squared_over_sigma_y_squared - (sum_x_over_sigma_y_squared)**2
    
    # Calculate sigma_b^2
    sigma_b_squared = (1 / delta) * sum_inv_sigma_y_squared
    
    return sigma_b_squared**0.5




# Constants
c = 299792458  # Speed of light in m/s
e = 1.602e-19  # Elementary charge in C (Joule/eV)

# Given data
wavelengths = [390, 455, 505, 535, 590, 615, 640, 935]
Vstops_original = [1.21, 0.898, 0.7218, 0.597, 0.509, 0.3982, 0.3243, 0.4863]  # Original Vstop in volts
Vstop_Err = 0.0001 * e  # Error in J
print(Vstop_Err)
# Convert stopping voltage to electron volts (eV)
Vstops_eV = [v * e for v in Vstops_original]

# Convert wavelengths to frequencies
frequencies = [c / (wavelength * 1e-9) for wavelength in wavelengths]

# Exclude the outlier at 935 nm
frequencies_excl_outlier = frequencies[:-1]
Vstops_excl_outlier = Vstops_eV[:-1]

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(frequencies_excl_outlier, Vstops_excl_outlier)

# Generate the line of best fit
fit_line_x = np.linspace(min(frequencies_excl_outlier), max(frequencies_excl_outlier), 100)
fit_line_y = slope * fit_line_x + intercept

# Calculate residuals
predicted_Vstops = [slope * freq + intercept for freq in frequencies_excl_outlier]
residuals = np.subtract(Vstops_excl_outlier, predicted_Vstops)

# Plotting the main graph and residuals
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
fig.suptitle('Frequency vs. Stopping Voltage with Linear Fit and Residuals')

# Main plot with data points, error bars, and best fit line
ax1.errorbar(frequencies, Vstops_eV, yerr=Vstop_Err, fmt='o', capsize=4, label='Vstop ± Error')
ax1.plot(fit_line_x, fit_line_y, color='red', linestyle='--', label='Best Fit Line (Excl. 935 nm)')
ax1.set_ylabel('eVstop (J)')
ax1.grid(True)
ax1.legend()

# Residuals plot
ax2.errorbar(frequencies_excl_outlier, residuals, yerr=Vstop_Err, fmt='o', capsize=4)
ax2.axhline(0, color='red', linestyle='--')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Residuals (J)')
ax2.grid(True)

plt.show()

error = sigma_b(frequencies_excl_outlier, [Vstop_Err]*len(frequencies_excl_outlier))


# Output results
# Planck's constant in J
h = slope
print("Planck's constant (h):", h, u"\u00B1", error, "Js")

# Work function in eV
work_function_eV = -intercept
print("Work function (phi):", work_function_eV, "J")

# R value, STD error:
print("R value for fit: ", r_value, "with stderr: ", std_err)

# Chi-squared calculation
Vstop_errors = [Vstop_Err] * len(Vstops_excl_outlier)
chi_squared = np.sum(np.power(np.divide(np.subtract(predicted_Vstops, Vstops_excl_outlier), Vstop_errors), 2))
print("Chi-squared (χ²) value:", chi_squared)

#Part 2
intensities = [1, 2, 3, 4]
photocurrents = [-0.0829, -0.1042, -0.1446, -0.2162] # raw data in volts
R = 100e3 # ohms
photocurrents_err = 0.0001 / R
photocurrents = [abs(volt) / R for volt in photocurrents] # amps

# Plotting
plt.figure(figsize=(8, 6))
plt.errorbar(intensities, photocurrents, yerr=photocurrents_err, fmt='o', capsize=4, label='photocurrent ± Error')
plt.xlabel('Intensity Setting')
plt.ylabel('Photocurrent (Amps)')
plt.title('Photocurrent Based on Intensity Setting at Fixed Vstop')
plt.grid(True)
plt.legend()
plt.show()