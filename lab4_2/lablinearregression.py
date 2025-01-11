import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([-6,-5,-4,-3,-2,-1,0,1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([-392,-353,-321,-239,-191,-116,-51,75,154,179,228,282,365])

model = LinearRegression()
model.fit(x, y)


y_pred = model.predict(x)


residuals = y - y_pred

std_dev = np.std(residuals)

residual_variance = np.var(residuals)
std_dev_calc = np.sqrt(residual_variance)


slope = model.coef_[0]
intercept = model.intercept_


print("Slope of the fitted line:", slope)
print("Intercept of the fitted line:", intercept)
print("Residuals:", residuals)
print("Residual variance:", residual_variance)
print("Standard Deviation of residuals:", std_dev_calc)

plt.scatter(x, y, color='blue', label='Data Points')


plt.plot(x, y_pred, color='red', label='Fitted Line')

plt.fill_between(x.flatten(), y_pred - std_dev, y_pred + std_dev, color='gray', alpha=0.3, label="Standard Deviation Range")


plt.text(0.5, max(y_pred) - 1, f'Slope = {slope:.2f}\nStandard Deviation = {std_dev_calc:.2f}', color='black', fontsize=12)

plt.title('Fitted Line with Slope and Standard Deviation')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.show()
