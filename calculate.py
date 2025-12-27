import numpy as np
from numpy import linalg as ln

# Главное меню
# region 
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
# endregion 

# Меню матричного калькулятора
# region
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
# endregion

# Логика матричного калькулятора
# region
# Сложение матриц между собой
def addition_matrix():
    size = int(input("Введите размер квадратичной матрицы: "))

    # Проверка на наличие положительного числа
    if size <= 0 or not isinstance(size, int):
        print("Размер матрицы должен быть положительным числом")
    
    # Создаем пустые матрицы заданного размера
    matrix_A = np.zeros((size, size))
    matrix_B = np.zeros((size, size))

    # Запрашиваем у пользователя элементы матрицы А
    print("Введите элементы матрицы А {rows}x{cols} по порядку: ")
    for i in range(size):
        for j in range(size):
            matrix_A[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    # Запрашиваем у пользователя элементы матрицы B
    print("Введите элементы матрицы B {rows}x{cols} по порядку:")
    for i in range(size):
        for j in range(size):
            matrix_B[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    print("\nВаша матрица A {size}x{size}:")
    print(matrix_A)

    print("Ваша матрица В {size}x{size}:")
    print(matrix_B)

    print("Сложенная матрица А и В:")
    print(matrix_A + matrix_B)

    print("Сложенная матрица В и А:")
    print(matrix_B + matrix_A)

# Вычитание матриц между собой
def subtraction_matrix():
    size = int(input("Введите размер квадратичной матрицы: "))

    # Проверка на наличие положительного числа
    if size <= 0 or not isinstance(size, int):
        print("Размер матрицы должен быть положительным числом")
    
    # Создаем пустые матрицы заданного размера
    matrix_A = np.zeros((size, size))
    matrix_B = np.zeros((size, size))

    # Запрашиваем у пользователя элементы матрицы А
    print("Введите элементы матрицы А {rows}x{cols} по порядку: ")
    for i in range(size):
        for j in range(size):
            matrix_A[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    print("Введите элементы матрицы B {rows}x{cols} по порядку:")
    for i in range(size):
        for j in range(size):
            matrix_B[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    print("\nВаша матрица A {size}x{size}:")
    print(matrix_A)

    print("Ваша матрица В {size}x{size}:")
    print(matrix_B)

    print("Вычитание матрицы А и В:")
    print(matrix_A - matrix_B)

    print("Вычитание матрицы В и А:")
    print(matrix_B - matrix_A)

# Умножение матриц между собой
def multiplication_matrix():
    rows = int(input("Введите количество строк матрицы: "))
    cols = int(input("Введите количество столбцов матрицы: "))

    # Создаем пустые матрицы заданного размера
    matrix_A = np.zeros((rows, cols))
    matrix_B = np.zeros((rows, cols))

    # Запрашиваем у пользователя элементы матрицы А
    print("Введите элементы матрицы А {rows}x{cols} по порядку: ")
    for i in range(rows):
        for j in range(cols):
            matrix_A[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    print("Введите элементы матрицы B {rows}x{cols} по порядку:")
    for i in range(rows):
        for j in range(cols):
            matrix_B[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    print("\nВаша матрица A {rows}x{cols}:")
    print(matrix_A)

    print("Ваша матрица В {rows}x{cols}:")
    print(matrix_B)

    print(np.dot(matrix_A, matrix_B))

# Transponent Matrix
def transponent_matrix():
    # Запрашиваем у пользователя размер матриц
    rows = int(input("Введите оличество строк матрицы: "))
    cols = int(input("Введите количество столбцов матрицы: "))

    # Создаем пустую матрицу заданного размера
    matrix_A = np.zeros((rows, cols))

    # Запрашиваем у пользователя элементы матрицы А
    print("Введите элементы матрицы А {rows}x{cols} по порядку: ")
    for i in range(rows):
        for j in range(cols):
            matrix_A[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    # Выводим исходную матрицу
    print("Ваша исходная матрица: ")
    print(matrix_A)

    # Выводим полученную матрицу
    print("\nВаша транспонированная матрица: ")
    print(matrix_A.transpose())

# Вычисление определителя матрицы
def determinant_matrix():
    size = int(input("Введите размер квадратичной матрицы: "))

    # Проверка на наличие положительного числа
    if size <= 0 or not isinstance(size, int):
        print("Размер матрицы должен быть положительным числом")

    # Создаем пустую матрицу заданного размера
    matrix_A = np.zeros((size, size))

    # Запрашиваем у пользователя элементы матрицы А
    print("Введите элементы матрицы А {rows}x{cols} по порядку: ")
    for i in range(size):
        for j in range(size):
            matrix_A[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    Det_A = ln.det(matrix_A)

    print("\nВаша матрица A {size}x{size}:")
    print(matrix_A)

    print("\nВаш определитель матрицы A {size}x{size}:")
    print(int(Det_A))

# Вычисление обратной матрицы
def ff_matrix():
    size = int(input("Введите размер квадратичной матрицы: "))

    # Проверка на наличие положительного числа
    if size <= 0 or not isinstance(size, int):
        print("Размер матрицы должен быть положительным числом")

    # Создаем пустую матрицу заданного размера
    matrix_A = np.zeros((size, size))

    # Запрашиваем у пользователя элементы матрицы А
    print("Введите элементы матрицы А {rows}x{cols} по порядку: ")
    for i in range(size):
        for j in range(size):
            matrix_A[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    A_inv = ln.inv(matrix_A)

    print("\nВаша матрица A {size}x{size}:")
    print(matrix_A)

    print("\nВаша обратная матрица A {size}x{size}:")
    print(A_inv)

# Matrix Rank
def rank_matrix():
    # Запрашиваем у пользователя размер матриц
    rows = int(input("Введите оличество строк матрицы: "))
    cols = int(input("Введите количество столбцов матрицы: "))

    # Создаем пустую матрицу заданного размера
    matrix_A = np.zeros((rows, cols))

    # Запрашиваем у пользователя элементы матрицы А
    print("Введите элементы матрицы А {rows}x{cols} по порядку: ")
    for i in range(rows):
        for j in range(cols):
            matrix_A[i, j] = float(input(f"Элемент [{i+1}, {j+1}]: "))

    # Выводим исходную матрицу
    print("Ваша исходная матрица: ")
    print(matrix_A)

    # Выводим полученную матрицу
    print("\nВаш ранг матрицы: ")
    print(ln.matrix_rank(matrix_A))
# endregion

# Логика векторного калькулятора
# region
def input_matrix(name):
    print(f"\n{'=' * 50}")
    print(f"Ввод матрицы {name}")
    print('=' * 50)

    # Ввод размеров
    while True:
        try:
            sizes = input(f"Введите размеры матрицы {name} (строки столбцы через пробел): ").strip().split()
            rows, cols = map(int, sizes)
            if rows <= 0 or cols <= 0:
                raise ValueError("Размеры должны быть положительными числами.")
            break
        except ValueError as e:
            print(f"Ошибка: {e}. Попробуйте снова.")

    # Ввод данных
    print(f"Введите данные для матрицы {name} ({rows}x{cols}):")
    data = []
    for i in range(rows):
        while True:
            try:
                row = input(f"  Строка {i + 1} (через пробел): ").strip().split()
                if len(row) != cols:
                    raise ValueError(f"Строка должна содержать ровно {cols} элементов.")
                row = list(map(float, row))  # Используем float для универсальности
                data.append(row)
                break
            except ValueError as e:
                print(f"Ошибка: {e}. Попробуйте снова.")

    return np.array(data)\

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

def solve_algebraic(A, b):
    A_inv = np.linalg.inv(A)
    x = A_inv @ b
    return x

def solve_internal(A, b):
    x = np.linalg.solve(A, b)
    return x
# endregion

# Интерфейс векторов
# region
def vector():
    # Векторы
    v1 = input_matrix("A")
    v2 = input_matrix("B")
    print(f"Скалярное произведение {scalar_product(v1, v2)}")

    # Матрицы
    m1 = input_matrix("A")
    m2 = input_matrix("B")

    print(f"Сложение {add(m1, m2)}")
    print(f"Вычитание {sub(m1, m2)}")
    print(f"Умножение {mult(m1, m2)}")
    print(f"Транспонирование {T(m1)}")

    # Решение систем
    n = int(input("Введите количество уравнений и переменных (n): "))

    A = np.zeros((n, n))
    b = np.zeros(n)

    print("Введите коэффициенты матрицы A (строка за строкой, разделяя пробелами):")
    for i in range(n):
        row = list(map(float, input(f"Строка {i + 1}: ").split()))
        A[i] = row

    print("Введите вектор b (элементы через пробел):")
    b = np.array(list(map(float, input().split())))

    try:
        x_algebraic = solve_algebraic(A, b)
        print("Решение алгебраическим методом:", x_algebraic)

        x_internal = solve_internal(A, b)
        print("Решение внутренним методом:", x_internal)

        # Проверяем, совпадают ли решения (с небольшой погрешностью)
        if np.allclose(x_algebraic, x_internal):
            print("Решения совпадают.")
        else:
            print("Решения не совпадают (возможно, из-за численной погрешности).")
    except np.linalg.LinAlgError as e:
        print(f"Ошибка при решении: {e}")
# endregion

# Меню систем линейного уравнения
# region
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
# endregion

# Внутренний метод (с помощью готовых функций numpy)
def internal_method():
    print("ВНУТРЕННИЙ МЕТОД")

    n = int(input("Введите количество уравнений (размер системы): "))

    # Создание матрицы
    Ab = np.zeros((n, n + 1))

    # Наполнение матрицы
    print(f"\n[A|B] ({n}x{n + 1}):")
    for i in range(n):
        for j in range(n + 1):
            if j < n:
                value = float(input(f"A[{i + 1}, {j + 1}]: "))
                Ab[i, j] = value
            else:
                value = float(input(f"B[{i + 1}]: "))
                Ab[i, j] = value

    # Разделение целой матрицы Ab на отдельные части A и b
    A = Ab[:, :n]
    b = Ab[:, n]

    # Вывод исходной матрицы
    print("\nИсходная матрица [A|B]:")
    print_matrix_with_separator(Ab)

    # Вывод решения
    print("\nРешение: ")
    print(np.linalg.solve(A, b))

# Метод Крамера
def cramer_method():
    print("МЕТОД КРАМЕРА")

    n = int(input("Введите количество уравнений (размер системы): "))
    
    # Создаем матрицу коэффициентов A и вектор B
    A = np.zeros((n, n))
    B = np.zeros(n)
    
    print(f"\nВведите коэффициенты матрицы A ({n}x{n}):")
    for i in range(n):
        for j in range(n):
            A[i, j] = float(input(f"A[{i+1},{j+1}]: "))
    
    print(f"\nВведите правые части уравнений (вектор B):")
    for i in range(n):
        B[i] = float(input(f"B[{i+1}]: "))
    

    print("ИСХОДНАЯ СИСТЕМА:")
    for i in range(n):
        equation = f"{A[i,0]:.2f}x₁"
        for j in range(1, n):
            equation += f" + {A[i,j]:.2f}x{j+1}"
        equation += f" = {B[i]:.2f}"
        print(equation)
    
    # Вычисляем главный определитель
    det_A = ln.det(A)
    
    if abs(det_A) < 1e-10:
        print("\n⚠️ ОПРЕДЕЛИТЕЛЬ СИСТЕМЫ РАВЕН НУЛЮ!")
        print("Система либо не имеет решений, либо имеет бесконечно много решений")
        return
    
    print(f"\nΔ = det(A) = {det_A:.6f}")
    
    # Вычисляем определители для каждого неизвестного
    print("\nВычисление определителей для каждого неизвестного: ")
    solutions = []
    for j in range(n):
        A_j = A.copy()
        A_j[:, j] = B  # Заменяем j-тый столбец на вектор B
        det_A_j = ln.det(A_j)
        x_j = det_A_j / det_A
        solutions.append(x_j)
        print(f"Δ{j+1} = {det_A_j:.6f}, x{j+1} = Δ{j+1}/Δ = {x_j:.6f}")
    
    
    print("\nРЕШЕНИЕ СИСТЕМЫ:")
    for i, sol in enumerate(solutions):
        print(f"x{i+1} = {sol:.6f}")

# Метод обратной матрицы
def inverse_matrix_method():
    print("МЕТОД ОБРАТНОЙ МАТРИЦЫ")

    n = int(input("Введите количество уравнений (размер системы): "))
    
    # Создание матрицы
    A = np.zeros((n, n))
    B = np.zeros(n)
    
    # Наполнение матрицы А
    print(f"\nВведите коэффициенты матрицы A ({n}x{n}):")
    for i in range(n):
        for j in range(n):
            A[i, j] = float(input(f"A[{i+1},{j+1}]: "))
    
    # Наполнение вектора B
    print(f"\nВведите правые части уравнений (вектор B):")
    for i in range(n):
        B[i] = float(input(f"B[{i+1}]: "))
    
    # Вывод матрицы
    print("\nМатрица A:")
    print(A)
    print("\nВектор B:", B)
    
    try:
        # Проверяем обратимость матрицы
        det_A = ln.det(A)
        if abs(det_A) < 1e-10:
            print(f"\n❌ Матрица A вырождена! det(A) = {det_A:.6f}")
            print("Метод обратной матрицы не применим")
            return
        
        print(f"\ndet(A) = {det_A:.6f} ≠ 0 → матрица обратима")
        
        # Вычисляем обратную матрицу
        A_inv = ln.inv(A)
        print("\nОбратная матрица A⁻¹:")
        print(A_inv)
        
        # Решение: x = A⁻¹ · B
        solution = np.dot(A_inv, B)
        
        print("\nРЕШЕНИЕ СИСТЕМЫ (x = A⁻¹·B):")
        for i, sol in enumerate(solution):
            print(f"x{i+1} = {sol:.6f}")
        
    except ln.LinAlgError:
        print("\n❌ Ошибка: матрица сингулярна или плохо обусловлена")

# Метод Гаусса
def gauss_method():
    print("МЕТОД ГАУССА (ПРЯМОЙ ХОД)")
    
    n = int(input("Введите количество уравнений (размер системы): "))
    
    # Создаем расширенную матрицу [A|B]
    Ab = np.zeros((n, n+1))
    
    print(f"\nВведите коэффициенты расширенной матрицы [A|B] ({n}x{n+1}):")
    for i in range(n):
        for j in range(n+1):
            if j < n:
                Ab[i, j] = float(input(f"A[{i+1},{j+1}]: "))
            else:
                Ab[i, j] = float(input(f"B[{i+1}]: "))
    
    print("\nИсходная расширенная матрица [A|B]:")
    print_matrix_with_separator(Ab)
    
    # Прямой ход метода Гаусса
    for i in range(n):
        # Поиск главного элемента в столбце
        max_row = i
        for k in range(i+1, n):
            if abs(Ab[k, i]) > abs(Ab[max_row, i]):
                max_row = k
        
        # Перестановка строк
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
            print(f"\nПерестановка строк {i+1} и {max_row+1}:")
            print_matrix_with_separator(Ab)
        
        # Проверка на нулевой диагональный элемент
        if abs(Ab[i, i]) < 1e-10:
            print(f"\n⚠️ Нулевой диагональный элемент в строке {i+1}")
            print("Система вырождена или требует особого подхода")
            return
        
        # Нормировка текущей строки
        pivot = Ab[i, i]
        if abs(pivot - 1.0) > 1e-10:
            Ab[i] = Ab[i] / pivot
            print(f"\nНормировка строки {i+1} (деление на {pivot:.3f}):")
            print_matrix_with_separator(Ab)
        
        # Исключение переменной из последующих строк
        for k in range(i+1, n):
            factor = Ab[k, i]
            if abs(factor) > 1e-10:
                Ab[k] = Ab[k] - factor * Ab[i]
                print(f"\nИсключение x{i+1} из строки {k+1} (вычитание {factor:.3f}×строка{i+1}):")
                print_matrix_with_separator(Ab)
    
    # Обратный ход
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = Ab[i, n]  # Правая часть
        for j in range(i+1, n):
            x[i] -= Ab[i, j] * x[j]
        x[i] /= Ab[i, i]
    
    print("\nТРЕУГОЛЬНАЯ МАТРИЦА И РЕШЕНИЕ:")
    print_matrix_with_separator(Ab)
    
    print("\nРЕШЕНИЕ СИСТЕМЫ:")
    for i, sol in enumerate(x):
        print(f"x{i+1} = {sol:.6f}")
    
    # Проверка
    A = Ab[:, :n]
    B = Ab[:, n]
    check = np.dot(A, x) - B
    print(f"\nНорма невязки: {ln.norm(check):.2e}")

# Метод LU-Разложение
def LU():
    print("LU-РАЗЛОЖЕНИЕ")

    n = int(input("Введите количество уравнений (размер системы): "))        
    
    # Создание основной матрицы A (n x n)
    print("\nВведите матрицу A размером {n}x{n}:")
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = float(input(f"A[{i+1},{j+1}] = "))
    
    # Создание вектора свободных членов b (n x n)
    print("\nВведите вектор свободных членов b:")
    b = np.zeros(n)
    for i in range(n):
        b[i] = float(input(f"b[{i+1}] = "))
    
    # Инициализация матриц L и U
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Копируем A в U для преобразований
    U = A.copy().astype(float)
    
    # Инициализируем L как единичную матрицу
    for i in range(n):
        L[i, i] = 1.0

    print("Начальные матрицы:")
    print("A =\n", A)
    print("b =", b)

    # Прямой ход метода Гаусса для получения U и L
    for k in range(n-1):  # Для каждого столбца кроме последнего
        # Проверка на нулевой ведущий элемент
        if abs(U[k, k]) < 1e-10:
            print(f"Внимание! Ведущий элемент U[{k+1},{k+1}] ≈ 0")
            print("Нужна перестановка строк (PLU-разложение)")
            # Ищем ненулевой элемент ниже
            for i in range(k+1, n):
                if abs(U[i, k]) > 1e-10:
                    # Меняем строки в U
                    U[[k, i]] = U[[i, k]]
                    # Меняем строки в L (но только до k-1 столбца!)
                    if k > 0:
                        L[[k, i], :k] = L[[i, k], :k]
                    # Меняем элементы в b
                    b[[k, i]] = b[[i, k]]
                    print(f"Переставили строки {k+1} и {i+1}")
                    break
        
        for i in range(k+1, n):  # Для строк ниже текущей
            if U[k, k] != 0:
                # Вычисляем множитель
                multiplier = U[i, k] / U[k, k]
                # Сохраняем множитель в L
                L[i, k] = multiplier
                
                # Обнуляем элемент в U и вычитаем из строки
                for j in range(k, n):
                    U[i, j] -= multiplier * U[k, j]
    
    print("\nРЕЗУЛЬТАТ LU-РАЗЛОЖЕНИЯ:")
    print("Матрица L (нижняя треугольная):")
    print(np.round(L, 4))
    
    print("\nМатрица U (верхняя треугольная):")
    print(np.round(U, 4))
    
    # Решение системы
    print("РЕШЕНИЕ СИСТЕМЫ:")
    
    # 1. Решаем Ly = b (прямая подстановка)
    y = np.zeros(n)
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val += L[i, j] * y[j]
        y[i] = b[i] - sum_val  # L[i,i] = 1, поэтому не делим
    
    print("Промежуточный вектор y (из Ly = b):")
    print_vector(y)

    # 2. Решаем Ux = y (обратная подстановка)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):  # Снизу вверх
        sum_val = 0
        for j in range(i+1, n):
            sum_val += U[i, j] * x[j]
        x[i] = (y[i] - sum_val) / U[i, i]
    
    print("\nРешение x (из Ux = y):")
    print_vector(x)

    # Проверка решения
    Ax = np.dot(A, x)
    print("\nПроверка: A * x =")
    print_vector(Ax)
    print("Оригинальный вектор b:")
    print_vector(b)
    
    residual = np.max(np.abs(Ax - b))
    print(f"\nНевязка решения: {residual:.10f}")
    
    # Определитель
    det_A = np.prod(np.diag(U))  # det(L) = 1
    print(f"\nОпределитель матрицы A: det(A) = {det_A:.6f}")
    
    return L, U, x

# МНК
def mnk():
    print("МЕТОД НАИМЕНЬШИХ КВАДРАТОВ (МНК)")
    print("=" * 50)
    
    # Ввод размерности
    m = int(input("Количество уравнений (строк матрицы A): "))
    n = int(input("Количество неизвестных (столбцов матрицы A): "))
    
    # Ввод матрицы A
    print(f"\nВведите матрицу A размером {m}x{n} (по {n} чисел в строке):")
    A = []
    for i in range(m):
        while True:
            try:
                row_input = input(f"Строка {i+1}: ")
                row = list(map(float, row_input.split()))
                if len(row) != n:
                    print(f"Ошибка: нужно {n} чисел, а вы ввели {len(row)}!")
                    continue
                A.append(row)
                break
            except ValueError:
                print("Ошибка: введите числа через пробел!")
    
    # Ввод вектора b
    print(f"\nВведите вектор b размером {m}:")
    while True:
        try:
            b_input = input("Элементы через пробел: ")
            b = list(map(float, b_input.split()))
            if len(b) != m:
                print(f"Ошибка: нужно {m} чисел, а вы ввели {len(b)}!")
                continue
            break
        except ValueError:
            print("Ошибка: введите числа через пробел!")
    
    A = np.array(A)
    b = np.array(b)
    
    # Решение МНК
    x = np.linalg.pinv(A) @ b
    
    # Вывод результатов
    print("\n" + "=" * 50)
    print("ВВЕДЕННЫЕ ДАННЫЕ:")
    print(f"A ({m}x{n}) =")
    print(A)
    print(f"\nb ({m}) = {b}")
    
    print("\n" + "=" * 50)
    print("РЕШЕНИЕ:")
    print(f"x = {x}")
    
    # Проверка
    y_pred = A @ x
    residual = np.linalg.norm(y_pred - b)
    
    print(f"\nПроверка:")
    print(f"Ax = {y_pred}")
    print(f"b  = {b}")
    print(f"Невязка ||Ax - b|| = {residual:.10f}")
    
    # Детальная проверка
    print(f"\nДетальная проверка:")
    for i in range(m):
        print(f"Точка {i+1}: Ax[{i}] = {y_pred[i]:.6f}, b[{i}] = {b[i]}, разница = {y_pred[i]-b[i]:.6f}")
    
    return x

# Вспомогательные функции
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

def print_vector(v):
    """Красивый вывод вектора"""
    print("[", end="")
    for i, val in enumerate(v):
        if i > 0:
            print(", ", end="")
        print(f"{val:7.3f}", end="")
    print("]")