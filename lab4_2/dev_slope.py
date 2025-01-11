import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data points
x = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([-392, -353, -321, -239, -191, -116, -51, 75, 154, 179, 228, 282, 365])

# Fit the linear regression model
model = LinearRegression()
model.fit(x, y)

# Predict y values
y_pred = model.predict(x)

# Calculate residuals
residuals = y - y_pred

# Calculate residual variance
residual_variance = np.var(residuals, ddof=1)  # ddof=1 for unbiased estimate

# Calculate sum of squares of x (Sxx)
x_mean = np.mean(x)
Sxx = np.sum((x - x_mean) ** 2)

# Calculate standard deviation of the slope
std_dev_slope = np.sqrt(residual_variance / Sxx)

# Extract slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

# Calculate upper and lower bound slopes
slope_upper = slope + std_dev_slope
slope_lower = slope - std_dev_slope

# Generate lines for upper and lower bound slopes
y_upper = slope_upper * x + intercept
y_lower = slope_lower * x + intercept

# Print results
print("Slope of the fitted line:", slope)
print("Intercept of the fitted line:", intercept)
print("Residual variance:", residual_variance)
print("Standard Deviation of the slope:", std_dev_slope)

# Plot data points and fitted line
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', label='Fitted Line')

# Highlight deviation of slope
plt.plot(x, y_upper, '--', color='green', label='Upper Bound (Slope + Std Dev)')
plt.plot(x, y_lower, '--', color='orange', label='Lower Bound (Slope - Std Dev)')

# Add a text annotation to display slope and its standard deviation
plt.text(
    0.5, max(y_pred) - 20,
    f'Slope = {slope:.2f}\nStd Dev of Slope = {std_dev_slope:.2f}',
    color='black', fontsize=12
)

# Add titles and labels
plt.title('Fitted Line with Highlighted Slope Deviation')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
