
from config import *
from agent import Agent
from salepoint import SalePoint
from copy import deepcopy
from random import shuffle, sample
import numpy as np



class Manager:

    def __init__(self, times_train: list[int], times_test: list[int]):

        # check
        assert N_ITEMS % SIZE_BATCH == 0, "ERROR: N_ITEMS % SIZE_BATCH != 0"
        
        self.count_iters = N_ITEMS // SIZE_BATCH

        # SALEPOINTS
        self.TRAIN_SP = [ SalePoint(t) for t in times_train ]
        self.TEST_SP  = [ SalePoint(t) for t in times_test ]
        
        # calc size_input
        sp = self.TRAIN_SP[0]
        item = sp.item_matrix[0]
        SIZES_NET[0] = sp.len_inp_1_item(item) #
        
        # arrays AGENTS
        self.POPUL = []
        for i in range(N_POPUL):
            ag = Agent(SIZES_NET)
            ag._init_random_net()
            self.POPUL.append(ag)
        self.PARENTS = []
        self.CHILDS  = []
        self.ELITE   = []

        self.BEST_TRAIN = [None]
        self.BEST_TEST  = [None]

        self.results = None

        self.iter = None
        self.total_iters = None


    def _calc_reward (self, agent: Agent):
        # расчет награды
        # check
        assert agent.total_rew is None

        agent.total_rew = (agent.arr_rew['rew_sales'] - agent.arr_rew['rew_neud_sum']) / agent.arr_rew['rew_stock']


    def _progon (self, arr_SP: list[SalePoint], arr_AG: list[Agent]):
        # прогон всех агентов по всем SalePoints
        # Agent RESET here
        for agent in arr_AG:
            agent.reset_state()
            arr_sp_cp = list(map(deepcopy, arr_SP))
            for sp in arr_sp_cp:

                assert sp.used == 0 # если эта sp отработана, то ее нельзя больше использовать

                for day in range(TIME_WORK):
                    sp.one_day()
                    for iter in range(self.count_iters):
                        inp = sp.make_batch_inp (SIZE_BATCH, iter)
                        zakaz = agent.think_batch (SIZE_BATCH, inp)
                        sp.put_zakaz_batch (zakaz, SIZE_BATCH, iter)
                agent.arr_rew ['rew_neud_sum'] += sp.rew_neud_sum
                agent.arr_rew ['rew_neud_num'] += sp.rew_neud_num
                agent.arr_rew ['rew_sales'   ] += sp.rew_sales 
                agent.arr_rew ['rew_stock'   ] += sp.rew_stock
                sp.used = 1 # эта sp отработана. ее нельзя больше использовать
            self._calc_reward (agent)
    

    def progon_train (self):
        # обучение
        self._progon (self.TRAIN_SP, self.POPUL)

        self.results = [round(ag.total_rew,3) for ag in self.POPUL]
        avg = sum(self.results) / len(self.results)
        mi = min(self.results)
        ma = max(self.results)
        
        print(f'[{self.iter: {self.len_i}}][train] progon all agents: avg = {avg:.2f}, min = {mi:.2f}, max = {ma:.2f}')


    def progon_test (self, li_test_ag: list[Agent]):
        # test
        self._progon (self.TEST_SP, li_test_ag)
        print(f'[{self.iter: {self.len_i}}][TEST]: total_rew = {li_test_ag[0].total_rew:.3f} | result of test SP: {li_test_ag[0].arr_rew}')

    
    def selection (self):
        # сначала отбор элит (1)
        # мутация 2 копий элит 
        # отбор турниром разм = 3 (совет Gemma4)

        # check
        assert len(self.POPUL)   == N_POPUL, "ERROR: len(POPUL) != N_POPUL"
        assert len(self.PARENTS) == 0      , "ERROR: PARENTS not empty"
        assert len(self.CHILDS)  == 0      , "ERROR: CHILDS not empty"
        assert len(self.ELITE)   == 0      , "ERROR: ELITE not empty"

        # ELITE
        self.POPUL.sort(key = lambda x:  x.total_rew) # сортировка
        self.elite = self.POPUL.pop(-1) # забираем лучшего

        # save best train
        self.save_best ([self.elite], self.BEST_TRAIN)
        
        self.ELITE.append (self.elite) # добавл элит
        for _ in range(2):
            copy_elite = deepcopy(self.elite)
            self._mut_agent (copy_elite)
            self.ELITE.append (copy_elite)

        # худший, с ним скрестим лучшего
        worst = self.POPUL.pop(0)
        self.ELITE.append (self._cross_pair(self.elite, worst, KOEFF_CROSS_EL_WO))  # [0.7,0.3]

        assert len(self.ELITE) == 4

        # отбор остальных
        # формирование пула родителей
        shuffle (self.POPUL)
        while self.POPUL:
            tmp = []
            for i in range(SIZE_TOUR):
                tmp.append (self.POPUL.pop(-1))
            tmp.sort(key=lambda x: x.total_rew)
            self.PARENTS.append (tmp[-1])

        # check
        assert len(self.POPUL) == 0


    def _cross_pair (self, par1, par2, koeff=[0.5, 0.5]):
        # усреднение весов
        # par1, par2: родители
        # koeff: коэф усреднения

        k1, k2 = koeff
        child = Agent(SIZES_NET) # agent.net = []
        for d1, d2 in zip(par1.net, par2.net): # d = {'W': W, 'b': b, 'FA': FA}
            if k1 == k2:
                W = (d1['W'] + d2['W']) / 2.0
                b = (d1['b'] + d2['b']) / 2.0
            else:
                W = k1*d1['W'] + k2*d2['W']
                b = k1*d1['b'] + k2*d2['b']

            assert W.shape == d1['W'].shape
            assert b.shape == d1['b'].shape

            FA = d1['FA']

            child.net.append ({'W': W, 'b': b, 'FA': FA})

        assert len(child.net) == len(par1.net)

        return child
    

    def cross (self):
        # cross many agents from PARENTS
        # agent.net = [{'W': W, 'b': b, 'FA': FA}, ]

        need_len = N_POPUL - len(self.ELITE)
        while len(self.CHILDS) < need_len:
            x, y = sample(self.PARENTS, k=2) # гарантия что x != y
            z = self._cross_pair (x, y) # with koeff=[0.5, 0.5]
            self.CHILDS.append(z)
        self.PARENTS = [] # очищаю
        return None
    

    def _mutate_adaptive (self, weights: np.ndarray):
        # in-place mutation

        S = weights.shape
        progress = self.iter / self.total_iters
        current_sigma = MAX_SIGMA_MUT - progress * (MAX_SIGMA_MUT - MIN_SIGMA_MUT)
        if current_sigma < 0: current_sigma = MIN_SIGMA_MUT 

        mutation_mask = np.random.rand(*S) < PROB_MUT
        perturbations = np.random.normal(loc=0.0, scale=current_sigma, size= S)

        weights[mutation_mask] += perturbations[mutation_mask]
        np.clip(weights, MIN_WEIGHT, MAX_WEIGHT, out=weights) 


    def _mut_agent (self, agent: Agent):
        # mut 1 agent
        for d in agent.net:
            W = d['W']
            b = d['b']
            self._mutate_adaptive (W)
            self._mutate_adaptive (b)


    def mutation (self):
        # mut many agents
        # mut only CHILDS, not ELITE

        for agent in self.CHILDS:
            self._mut_agent(agent)

        assert len(self.POPUL) == 0

        self.POPUL = self.CHILDS[:]  # копирую ссылки на агентов
        self.POPUL.extend (self.ELITE) # копирую ссылки на агентов
        
        self.CHILDS = [] # очищаю
        self.ELITE  = [] 

        assert len(self.POPUL) == N_POPUL


    def save_best (self, li_ag_new: list[Agent], li_ag_saved: list[Agent]):
        #
        if len(li_ag_new) > 1 or len(li_ag_saved) > 1:
            raise "ERROR: [save_best] len(li_ag_new) > 1 or len(li_ag_saved) > 1"
        
        ag_new = li_ag_new[0] # не изменяю, только читаю
        
        if li_ag_saved[0] is None:
            li_ag_saved[0] = deepcopy (ag_new)
        else:
            if ag_new.total_rew > li_ag_saved[0].total_rew:
                li_ag_saved[0] = deepcopy (ag_new)

    
    def run (self, N):
        # N: total_iters

        self.total_iters = N # it is used in _mutate_adaptive()
        self.len_i = len(str(N)) # for f print
        
        for i in range(N):

            self.iter = i # it is used in _mutate_adaptive()

            self.progon_train()
            self.selection()
            self.cross()
            self.mutation()

            if i and i % 2 == 0:
                self.progon_test(self.BEST_TRAIN)
                self.save_best (self.BEST_TRAIN, self.BEST_TEST)