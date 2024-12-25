import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

x = [30, 50, 60, 80, 90]
y = [37, 42, 57, 58, 65]

# Создание сплайна для сглаживания кривой
x_smooth = np.linspace(min(x), max(x), 150)
spline = interp1d(x, y, kind = 'linear')
y_smooth = spline(x_smooth)

plt.plot(x_smooth, y_smooth, label='Сглаженная кривая')
plt.plot(x, y, 'o', label='Исходные точки')

plt.title('График кривой зависимости уверенности распознавания от объёма датасета')
plt.xlabel('Число фото одного человека, шт')
plt.ylabel('Уверенность, %')

plt.grid(True)

plt.legend()

plt.show()

