
#from config import *
import numpy as np
    

class Agent:

    # важно!
    # на вход think подается вектор сразу из size_batch (возможно = 1) items (для эффективности расчета)
    # на выходе соответственно size_batch количеств заказа

    def __init__(self, sizes_net: list[int]):
        # sizes_net: лист вх размеров слоев + размер выхода [5, 10, 6, 1] -> [(5,10), (10,6), (6,1)]
        
        #chek
        assert min(sizes_net) > 0, "ERROR: min(sizes_net) <= 0"
        
        self.sizes_net = sizes_net
        self._make_struct()

        self.net = []
        self.reset_state()

    def reset_state (self):
        self.arr_rew = {'rew_neud_sum': 0, 'rew_neud_num': 0, 'rew_sales': 0, 'rew_stock': 0}
        self.total_rew = None

    def _make_struct (self):
        # создаем описание размеров структуры
        #  [5, 10, 6, 1] -> [(5,10), (10,6), (6,1)]
        self.struct = []
        for m,n in zip(self.sizes_net[:-1], self.sizes_net[1:]):
            self.struct.append((m,n))

    def _init_random_net(self):
        # создание векторов и матриц, и заполнение их случайными значениями
        for mn in self.struct:
            m,n = mn
            W = np.random.randn(m, n) * np.sqrt(2 / m) # совет Gemma4
            b = np.zeros((1, n)) # почему нулями? # совет Gemma4
            FA = np.tanh
            self.net.append ({'W': W, 'b': b, 'FA': FA})
        self.net[-1]['FA']  = self.ReLU  # после посл слоя

    def ReLU (self, x):
        # возвращает неотриц знач
        # используется после последнего слоя
        return np.maximum(0,x)
    
    def think_batch (self, size_batch: int, input: list[list[int]]):
        # прохождение данных через нейросеть
        # на входе батч из rows строк size_inp длиной каждая | shape = (rows, size_inp)
        # строка - данные одного item
        # на выходе сразу rows знач заказов | shape =(rows,1)

        X = np.array(input) #, dtype=np.float64)
        #X.shape = (1,-1)

        #check
        assert X.shape == (size_batch, self.sizes_net[0]), print(X.shape) #"ERROR: X.shape != (size_batch, self.sizes_net[0])"
        
        for layer in self.net:
            Z = np.dot(X, layer['W']) + layer['b']
            X = layer['FA'](Z)

        return [int(x[0]) for x in X] # size_batch zakaz для пополнения
    

if __name__ == '__main__':

    ex = Agent([10, 32, 1])