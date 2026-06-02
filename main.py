
from config import MAX_TIME_TRANSIT
from manager import Manager



def main():

    # задание времени поставки (дни) для точек продаж (SP)
    times_train = [1,3,5]
    times_test  = [2,4]

    # задание кол-во циклов обучения
    count_iters = 10

    assert max(times_train + times_test) <= MAX_TIME_TRANSIT

    man = Manager (times_train, times_test)
    man.run(count_iters)

    print('\nfinal result witn BEST_TRAIN')
    man.progon_test (man.BEST_TRAIN)
    
    print('\nfinal result witn BEST_TEST')
    man.progon_test (man.BEST_TEST)


main()