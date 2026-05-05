
from string import digits, ascii_letters
from math import pow
from random import choices


def decompose (num, base, len_arr):
    # num:     число которое надо разложить на разряды
    # base:    база исчисления (макс знач в разряде)
    # len_arr: требуемая длина массива результата (кол разрядов)

    arr = [0] * len_arr
    i = -1
    while True:
        arr[i] = num % base
        num = num // base
        if num == 0:
            break
        i -= 1
    return arr


def full_combo (len_s, simbols, max_count):

    len_simbols = len(simbols)
    res = []
    for n in range(max_count):
        ind = decompose (n, len_simbols, len_s)
        s = ''.join ([simbols[i] for i in ind])
        res.append(s)

    return res


def generate_unique_strings (length, count) -> list[str]:

    # строка может иметь повторы символов
    # но строки должны быть уникальны 

    ret = set()
    simbols = digits + ascii_letters
    len_simbols = len(simbols)

    try:
        max_count = int(pow (len_simbols, length))
    except Exception as E:
        print (f'ERROR: превышение вычислительных лимитов: "{E}"')
        return []

    if count > max_count:
        print (f'ERROR: невозможно сгенерировать {count} записей. Макс возможное кол-во: {max_count}')
        return []

    # если требуемое кол строк равно теоретическому, то просто перебрать все комбинации (это быстрее)
    if count == max_count:
        return full_combo (length, simbols, max_count)
    
    # TODO
    # проверить будет ли быстрее: при count близком к max_count генерить full_combo с max_count
    # а потом просто выбрать count готовых строк

    # иначе генерим случайные строки
    while len(ret) < count:
        list_s = choices (simbols, k= length)
        s = ''.join(list_s)
        ret.add (s)

    return list(ret)



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