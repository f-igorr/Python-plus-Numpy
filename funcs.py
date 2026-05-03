
import random
from string import ascii_letters, digits
from config import *




def generate_unique_strings(length: int, count: int) -> list[str]:
    """
    Генерирует заданное количество уникальных случайных строк заданной длины,
    с предварительной проверкой на возможность генерации.

    Строки состоят из строчных и заглавных латинских букв (string.ascii_letters) 
    и цифр (string.digits).

    :param length: Желаемая длина каждой генерируемой строки.
    :param count: Количество уникальных строк, которое необходимо сгенерировать.
    :return: Список (list) уникальных случайных строк или None при невозможности генерации.
    """

    characters = ascii_letters + digits
    num_chars = len(characters) # 62 символа

    try:
        max_possible = pow(num_chars, length)
    except OverflowError:
        print("Ошибка: Максимально возможное количество комбинаций превышает вычислительные лимиты.")
        return []

    if count > max_possible:
        print(f"ОШИБКА ГЕНЕРАЦИИ:")
        print(f"Невозможно сгенерировать {count} уникальных строк длиной {length}.")
        print(f"Максимальное возможное количество комбинаций при этих параметрах составляет: {max_possible:_}")
        return []

    # Инициализация для генерации
    unique_strings = set()
    
    # Устанавливаем лимит попыток, хотя проверка выше должна предотвратить бесконечный цикл
    max_attempts = count * 10 # Если комбинаций много, это может быть больше
    attempts = 0
    
    while len(unique_strings) < count and attempts < max_attempts:
        random_string = ''.join(random.choice(characters) for _ in range(length))
        unique_strings.add(random_string)
        attempts += 1

    if len(unique_strings) < count:
        print("\Генерация остановлена из-за внутреннего лимита попыток.")

    return list(unique_strings)




if __name__ == '__main__':

    # (length, count)
    
    x1 = (8, 5)           # ok
    x2 = (4, 200_000_000) # Слишком много строк
    x3 = (2, 4000)        # теор невозможно 

    X = (x1, x2, x3)

    for x in X:
        print(f'\n=== test length={x[0]}, count={x[1]} ===')
        results = generate_unique_strings(*x)
        assert len(set(results)) == len(results), "Ошибка: строки не являются уникальными!"
        if results:
            for i, s in enumerate(results):
                print(f"Строка {i+1}: {s}")