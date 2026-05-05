import numpy as np
from config import *
    

class Agent:

    # важно!
    # на вход think подается вектор сразу из size_batch (возможно = 1) items (для эффективности расчета)
    # на выходе соответственно size_batch количеств заказа

    def __init__(self, sizes_net: list[int]):
        # sizes_net: лист вх размеров слоев + размер выхода [5, 10, 6, 1] -> [(5,10), (10,6), (6,1)]
        
        assert None not in sizes_net, "ERROR: None in sizes_net !"
        
        self.sizes_net = sizes_net
        self._make_struct()

        self.net = []
        self.reset_state()

    def reset_state (self):
        self.arr_rew = {'rew_neud_sum': 0, 'rew_neud_num': 0, 'rew_sales': 0, 'rew_stock': 0}
        self.total_rew = None

    def _make_struct (self):
        # создаем описание размеров структуры
        # пример [5, 10, 6, 1] -> [(5,10), (10,6), (6,1)]
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
        self.net[-1]['FA']  = self._ReLU  # после посл слоя

    def _ReLU (self, x):
        # возвращает неотриц знач
        # используется после последнего слоя
        return np.maximum(0,x)
    
    def think_batch (self, size_batch: int, input: list[list[int]]):
        # прохождение данных через нейросеть
        # на входе батч из rows строк size_inp длиной каждая | shape = (rows, size_inp)
        # строка - данные одного item
        # на выходе сразу rows знач заказов | shape =(rows,1)

        X = np.array(input)

        assert X.shape == (size_batch, self.sizes_net[0]), print(X.shape)
        
        for layer in self.net:
            Z = np.dot(X, layer['W']) + layer['b']
            X = layer['FA'](Z)

        return [int(x[0]) for x in X] # size_batch zakaz для пополнения
    
    def set_reward (self, rew):
        self.total_rew = rew

    def add_arr_rew (self, key, val):
        self.arr_rew [key] += val

    def _mutate_adaptive (self, weights: np.ndarray, progress):
        # in-place mutation

        S = weights.shape
        current_sigma = MAX_SIGMA_MUT - progress * (MAX_SIGMA_MUT - MIN_SIGMA_MUT)
        
        if current_sigma < 0: 
            current_sigma = MIN_SIGMA_MUT 

        mut_mask = np.random.rand(*S) < PROB_MUT
        perts = np.random.normal(loc=0.0, scale=current_sigma, size= S)

        weights[mut_mask] += perts[mut_mask]
        np.clip(weights, MIN_WEIGHT, MAX_WEIGHT, out=weights) 

    def mut_net (self, iter, total_iters):
        # mutation of net
        progress = iter / total_iters
        for d in self.net:
            W = d['W']
            b = d['b']
            self._mutate_adaptive (W, progress)
            self._mutate_adaptive (b, progress)
    

if __name__ == '__main__':

    ex = Agent([10, 32, 1])