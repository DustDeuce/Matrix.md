# Матричный калькулятор на Python

## Структура проекта

### Главное меню
```python
def main_menu():
    print("\nВыбор раздела: ")
    
    print("1) Матричные преобразования (калькулятор)")
    print("2) Вектора")
    print("3) Решение систем линейных уравнений")
    print("\n0) Выход")

    command = int(input("\nВыберите метод матричного счисления: "))

    match command:
        case 1:
            matrix_menu()
        case 2:
            vector()
        case 3:
            linear_systems_menu()
```

### Меню матричного калькулятора
```python
def matrix_menu():
    print("Выбор счисления: ")

    print("\n1) Сложение")
    print("2) Вычитание")
    print("3) Умножение")
    print("4) Транспонирование")
    print("5) Вычисление определителя матрицы")
    print("6) Обратная матрица")
    print("7) Ранг матрицы")
    print("\n0) Назад")

    command = int(input("\nВыберите метод матричного счисления: "))

    match command:
        case 1:
            addition_matrix()
        case 2:
            subtraction_matrix()
        case 3:
            multiplication_matrix()    
        case 4:
            transponent_matrix()
        case 5:
            determinant_matrix()
        case 6:
            ff_matrix()
        case 7:
            rank_matrix()
        case 0:
            main_menu()
```

## Основные функции

### 1. Сложение матриц
```python
def addition_matrix():
    size = int(input("Введите размер квадратичной матрицы: "))

    if size <= 0 or not isinstance(size, int):
        print("Размер матрицы должен быть положительным числом")
    
    matrix_A = np.zeros((size, size))
    matrix_B = np.zeros((size, size))

    # Ввод матрицы A
    print("Введите элементы матрицы А {rows}x{cols} по порядку: ")
    for i in range(size):
        for j in range(size):
            matrix_A[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    # Ввод матрицы B
    print("Введите элементы матрицы B {rows}x{cols} по порядку:")
    for i in range(size):
        for j in range(size):
            matrix_B[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    print("\nРезультат сложения:")
    print("A + B =")
    print(matrix_A + matrix_B)
```

### 2. Вычитание матриц
```python
def subtraction_matrix():
    size = int(input("Введите размер квадратичной матрицы: "))

    if size <= 0 or not isinstance(size, int):
        print("Размер матрицы должен быть положительным числом")
    
    matrix_A = np.zeros((size, size))
    matrix_B = np.zeros((size, size))

    # Аналогичный ввод матриц
    # ...
    
    print("A - B =")
    print(matrix_A - matrix_B)
```

### 3. Умножение матриц
```python
def multiplication_matrix():
    rows = int(input("Введите количество строк матрицы: "))
    cols = int(input("Введите количество столбцов матрицы: "))

    matrix_A = np.zeros((rows, cols))
    matrix_B = np.zeros((rows, cols))

    # Ввод матриц
    # ...
    
    print("A × B =")
    print(np.dot(matrix_A, matrix_B))
```

### 4. Транспонирование
```python
def transponent_matrix():
    rows = int(input("Введите количество строк матрицы: "))
    cols = int(input("Введите количество столбцов матрицы: "))

    matrix_A = np.zeros((rows, cols))

    print("Введите элементы матрицы А {rows}x{cols} по порядку: ")
    for i in range(rows):
        for j in range(cols):
            matrix_A[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    print("\nТранспонированная матрица:")
    print(matrix_A.transpose())
```

### 5. Определитель матрицы
```python
def determinant_matrix():
    size = int(input("Введите размер квадратичной матрицы: "))

    if size <= 0 or not isinstance(size, int):
        print("Размер матрицы должен быть положительным числом")

    matrix_A = np.zeros((size, size))

    print("Введите элементы матрицы А {rows}x{cols} по порядку: ")
    for i in range(size):
        for j in range(size):
            matrix_A[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    Det_A = ln.det(matrix_A)

    print(f"\nОпределитель матрицы A {size}x{size}:")
    print(int(Det_A))
```

### 6. Обратная матрица
```python
def ff_matrix():
    size = int(input("Введите размер квадратичной матрицы: "))

    if size <= 0 or not isinstance(size, int):
        print("Размер матрицы должен быть положительным числом")

    matrix_A = np.zeros((size, size))

    print("Введите элементы матрицы А {rows}x{cols} по порядку: ")
    for i in range(size):
        for j in range(size):
            matrix_A[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    A_inv = ln.inv(matrix_A)

    print("\nОбратная матрица:")
    print(A_inv)
```

### 7. Ранг матрицы
```python
def rank_matrix():
    rows = int(input("Введите количество строк матрицы: "))
    cols = int(input("Введите количество столбцов матрицы: "))

    matrix_A = np.zeros((rows, cols))

    print("Введите элементы матрицы А {rows}x{cols} по порядку: ")
    for i in range(rows):
        for j in range(cols):
            matrix_A[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    print("\nРанг матрицы:")
    print(ln.matrix_rank(matrix_A))
```

## Векторные операции

### Ввод матрицы
```python
def input_matrix(name):
    print(f"\n{'=' * 50}")
    print(f"Ввод матрицы {name}")
    print('=' * 50)

    while True:
        try:
            sizes = input(f"Введите размеры матрицы {name} (строки столбцы через пробел): ").strip().split()
            rows, cols = map(int, sizes)
            if rows <= 0 or cols <= 0:
                raise ValueError("Размеры должны быть положительными числами.")
            break
        except ValueError as e:
            print(f"Ошибка: {e}. Попробуйте снова.")

    print(f"Введите данные для матрицы {name} ({rows}x{cols}):")
    data = []
    for i in range(rows):
        while True:
            try:
                row = input(f"  Строка {i + 1} (через пробел): ").strip().split()
                if len(row) != cols:
                    raise ValueError(f"Строка должна содержать ровно {cols} элементов.")
                row = list(map(float, row))
                data.append(row)
                break
            except ValueError as e:
                print(f"Ошибка: {e}. Попробуйте снова.")

    return np.array(data)
```

### Базовые операции
```python
def add(a, b):
    return np.add(a, b)

def sub(a, b):
    return np.subtract(a, b)

def mult(a, b):
    return a @ b

def T(a):
    return a.T

def scalar_product(a, b):
    return sum(i * j for i, j in zip(a, b))
```

## Решение систем линейных уравнений

### Меню методов решения
```python
def linear_systems_menu():
    print("РЕШЕНИЕ СИСТЕМ ЛИНЕЙНЫХ УРАВНЕНИЙ")
    
    print("\n1) Внутренний метод (через функции numpy)")
    print("2) Метод Крамера (через определители)")
    print("3) Метод обратной матрицы")
    print("4) Метод Гаусса (прямой ход)")
    print("5) LU-разложение")
    print("6) Метод наименьших квадратов")
    
    print("\n0) Назад в меню")
    
    command = int(input("\nВыберите метод решения: "))
    
    match command:
        case 1:
            internal_method()
        case 2:
            cramer_method()
        case 3:
            inverse_matrix_method()
        case 4:
            gauss_method()
        case 5:
            LU()
        case 6:
            mnk()
        case 0:
            matrix_menu()
```

### 1. Внутренний метод (numpy)
```python
def internal_method():
    print("ВНУТРЕННИЙ МЕТОД")

    n = int(input("Введите количество уравнений (размер системы): "))

    Ab = np.zeros((n, n + 1))

    print(f"\n[A|B] ({n}x{n + 1}):")
    for i in range(n):
        for j in range(n + 1):
            if j < n:
                value = float(input(f"A[{i + 1}, {j + 1}]: "))
                Ab[i, j] = value
            else:
                value = float(input(f"B[{i + 1}]: "))
                Ab[i, j] = value

    A = Ab[:, :n]
    b = Ab[:, n]

    print("\nРешение: ")
    print(np.linalg.solve(A, b))
```

### 2. Метод Крамера
```python
def cramer_method():
    print("МЕТОД КРАМЕРА")

    n = int(input("Введите количество уравнений (размер системы): "))
    
    A = np.zeros((n, n))
    B = np.zeros(n)
    
    print(f"\nВведите коэффициенты матрицы A ({n}x{n}):")
    for i in range(n):
        for j in range(n):
            A[i, j] = float(input(f"A[{i+1},{j+1}]: "))
    
    print(f"\nВведите правые части уравнений (вектор B):")
    for i in range(n):
        B[i] = float(input(f"B[{i+1}]: "))
    
    # Вычисляем главный определитель
    det_A = ln.det(A)
    
    if abs(det_A) < 1e-10:
        print("\n⚠️ ОПРЕДЕЛИТЕЛЬ СИСТЕМЫ РАВЕН НУЛЮ!")
        return
    
    # Вычисляем определители для каждого неизвестного
    solutions = []
    for j in range(n):
        A_j = A.copy()
        A_j[:, j] = B
        det_A_j = ln.det(A_j)
        x_j = det_A_j / det_A
        solutions.append(x_j)
    
    print("\nРЕШЕНИЕ СИСТЕМЫ:")
    for i, sol in enumerate(solutions):
        print(f"x{i+1} = {sol:.6f}")
```

### 3. Метод обратной матрицы
```python
def inverse_matrix_method():
    print("МЕТОД ОБРАТНОЙ МАТРИЦЫ")

    n = int(input("Введите количество уравнений (размер системы): "))
    
    A = np.zeros((n, n))
    B = np.zeros(n)
    
    # Ввод матрицы и вектора
    # ...
    
    try:
        det_A = ln.det(A)
        if abs(det_A) < 1e-10:
            print(f"\n❌ Матрица A вырождена!")
            return
        
        A_inv = ln.inv(A)
        solution = np.dot(A_inv, B)
        
        print("\nРЕШЕНИЕ СИСТЕМЫ (x = A⁻¹·B):")
        for i, sol in enumerate(solution):
            print(f"x{i+1} = {sol:.6f}")
        
    except ln.LinAlgError:
        print("\n❌ Ошибка: матрица сингулярна или плохо обусловлена")
```

### 4. Метод Гаусса
```python
def gauss_method():
    print("МЕТОД ГАУССА (ПРЯМОЙ ХОД)")
    
    n = int(input("Введите количество уравнений (размер системы): "))
    
    Ab = np.zeros((n, n+1))
    
    # Ввод данных
    # ...
    
    # Прямой ход метода Гаусса
    for i in range(n):
        # Поиск главного элемента
        max_row = i
        for k in range(i+1, n):
            if abs(Ab[k, i]) > abs(Ab[max_row, i]):
                max_row = k
        
        # Перестановка строк
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
        
        # Проверка на нулевой диагональный элемент
        if abs(Ab[i, i]) < 1e-10:
            print(f"\n⚠️ Нулевой диагональный элемент")
            return
        
        # Нормировка
        pivot = Ab[i, i]
        Ab[i] = Ab[i] / pivot
        
        # Исключение переменной
        for k in range(i+1, n):
            factor = Ab[k, i]
            Ab[k] = Ab[k] - factor * Ab[i]
    
    # Обратный ход
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = Ab[i, n]
        for j in range(i+1, n):
            x[i] -= Ab[i, j] * x[j]
        x[i] /= Ab[i, i]
    
    print("\nРЕШЕНИЕ СИСТЕМЫ:")
    for i, sol in enumerate(x):
        print(f"x{i+1} = {sol:.6f}")
```

### 5. LU-разложение
```python
def LU():
    print("LU-РАЗЛОЖЕНИЕ")

    n = int(input("Введите количество уравнений (размер системы): "))        
    
    # Ввод матрицы A и вектора b
    # ...
    
    # Инициализация матриц
    L = np.zeros((n, n))
    U = A.copy().astype(float)
    
    # L как единичная матрица
    for i in range(n):
        L[i, i] = 1.0

    # LU-разложение
    for k in range(n-1):
        for i in range(k+1, n):
            if U[k, k] != 0:
                multiplier = U[i, k] / U[k, k]
                L[i, k] = multiplier
                
                for j in range(k, n):
                    U[i, j] -= multiplier * U[k, j]
    
    # Решение системы
    # 1. Ly = b
    y = np.zeros(n)
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val += L[i, j] * y[j]
        y[i] = b[i] - sum_val
    
    # 2. Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum_val = 0
        for j in range(i+1, n):
            sum_val += U[i, j] * x[j]
        x[i] = (y[i] - sum_val) / U[i, i]
    
    return L, U, x
```

### 6. Метод наименьших квадратов
```python
def mnk():
    print("МЕТОД НАИМЕНЬШИХ КВАДРАТОВ (МНК)")
    
    # Ввод размерности
    m = int(input("Количество уравнений (строк матрицы A): "))
    n = int(input("Количество неизвестных (столбцов матрицы A): "))
    
    # Ввод матрицы A и вектора b
    # ...
    
    # Решение МНК
    x = np.linalg.pinv(A) @ b
    
    print(f"\nx = {x}")
    
    # Проверка
    y_pred = A @ x
    residual = np.linalg.norm(y_pred - b)
    
    print(f"Невязка ||Ax - b|| = {residual:.10f}")
    
    return x
```

## Вспомогательные функции

### Печать матрицы с разделителем
```python
def print_matrix_with_separator(matrix):
    """Печатает матрицу с разделителем для расширенной матрицы"""
    n = matrix.shape[1] - 1
    for i in range(matrix.shape[0]):
        row = ""
        for j in range(matrix.shape[1]):
            if j == n:
                row += " | "
            row += f"{matrix[i, j]:8.3f}"
        print(row)
```

### Печать вектора
```python
def print_vector(v):
    """Красивый вывод вектора"""
    print("[", end="")
    for i, val in enumerate(v):
        if i > 0:
            print(", ", end="")
        print(f"{val:7.3f}", end="")
    print("]")
```

## Запуск программы
```python
if __name__ == "__main__":
    main_menu()
```

## Требования
```python
import numpy as np
from numpy import linalg as ln
```
