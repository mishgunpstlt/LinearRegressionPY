import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Предположим, у нас есть данные о квартирах: площадь, этаж, расстояние до метро и до остановок общественного транспорта
X = np.array([
    [50, 2, 500, 300],    # площадь, этаж, расстояние до метро, расстояние до остановок
    [70, 3, 600, 400],
    [90, 1, 800, 200],
    [110, 4, 1000, 500],
    [130, 2, 700, 600],
    [150, 3, 900, 300],
    [170, 5, 1200, 400],
    [190, 1, 1000, 200],
    [210, 3, 1100, 700],
    [230, 2, 800, 500]
])

# Стоимость квартир (в млн рублей)
y = np.array([20, 30, 40, 45, 50, 55, 60, 65, 70, 75])

# Создаем объект множественной линейной регрессии и обучаем модель на тренировочных данных
model = LinearRegression()
model.fit(X, y)

# Получаем оценки параметров модели
intercept = model.intercept_
coefficients = model.coef_

print("Оценка параметров:")
print(f"Свободный член (intercept) = {intercept}")
print(f"Коэффициенты (coefficients) = {coefficients}")

# Предсказываем стоимость для тренировочных данных
y_pred = model.predict(X)

# Оцениваем качество модели
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Средняя квадратичная ошибка (MSE) = {mse}")
print(f"Коэффициент детерминации (R^2) = {r2}")

# Запрос ввода параметров для прогноза
print("\nВведите параметры для прогнозирования цены квартиры:")
area = float(input("Площадь квартиры, м²: "))
floor = int(input("Этаж квартиры: "))
dist_to_metro = int(input("Расстояние до метро, м: "))
dist_to_transport = int(input("Расстояние до остановок общественного транспорта, м: "))

# Прогнозируем цену для введенных параметров
input_data = np.array([[area, floor, dist_to_metro, dist_to_transport]])
predicted_price = model.predict(input_data)

print(f"\nПрогнозируемая цена для введенных параметров: {predicted_price[0]:.2f} млн. рублей")


# Визуализируем результаты
plt.figure(figsize=(10, 6))

# Для визуализации используем только один признак (площадь квартиры)
plt.scatter(X[:, 0], y, color='blue', label='Фактические значения')
plt.plot(X[:, 0], y_pred, color='red', linewidth=2, label='Предсказанные значения')

# Визуализация введенной точки
plt.scatter(area, model.predict([[area, floor, dist_to_metro, dist_to_transport]]), color='green', label='Введенная точка')

# Настройки графика
plt.xlabel('Площадь квартиры, м²')
plt.ylabel('Стоимость квартиры, млн. рублей')
plt.legend()
plt.title('Фактические, предсказанные значения и введенная точка')
plt.show()
