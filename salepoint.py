from random import uniform, normalvariate
from funcs import generate_unique_strings
from config import *
import numpy as np


class SalePoint:

    # TODO
    # заменить питоновские циклы вектоной обработкой numpy для ускорения расчетов.
    # сейчас мало используется векторная обработка данных
    # (например в симуляции торгового дня)

    def __init__(self, time_tr: int):
        # time_tr - время поставки

        self.time_tr = time_tr
        self.item_matrix = [] # ассортим матрица

        # метрики
        self.rew_neud_sum   = 0   # сумм неудовл спрос
        self.rew_neud_num   = 0   # кол неудовл спрос
        self.rew_sales      = 0   # сумм продажи
        self.rew_stock      = 0   # сумм склад

        self.used = 0 # флаг того что эта модель уже использовалась
        
        self._gen_assort_matrix()

    
    def set_used (self, flag):
        # флаг использовалась эта модель или нет [0 - не исп-сь, 1 исп-сь]
        
        assert flag != self.used
        
        self.used = flag


    def _calc_corr_k (self, mu: float, sigma: float, size: int = 10000) -> float:
        # расчет поправочного коэф для формулы: mu_fact * К = mu. 
        # нужен для возврата факт среднего сгенерированного множества к заданному
        # при обрезке отриц значений (замена на ноль).
        # для каждого набора (mu,sigma) своя поправка.

        summ = 0.0
        for i in range(size):
            summ += max(0, normalvariate(mu=mu, sigma=sigma))

        return mu * size / summ    
    

    def _gen_sales_per (self, mu: float, sigma: float, corr_k: float, per: int) -> int:
        # генерация продаж ЗА ПЕРИОД (напр. НЕДЕЛЯ) с поправкой на коэф corr_k
        # из-за замены отриц значений на ноль

        sales = 0
        for i in range(per):
            sales += round (corr_k * max(0, normalvariate(mu, sigma)))

        np.random.default_rng
        
        return sales


    def _gen_assort_matrix (self):
        # симуляция 3 групп товаров (A, B, C) (спрос: стабильный, средний, нестабильный)
        # с долями по кол-ву артикулов (0.6-0.75 , 0.15-0.25, 0.05-0.1) 
        # и с отношением sigma/mu (0.1-0.25, 0.2-0.4, 0.4-1.0)

        A_sku = round (uniform (0.6, 0.75) * N_ITEMS)
        A_k_sigma = (0.1, 0.25)

        C_sku = round (uniform (0.05, 0.1) * N_ITEMS)
        C_k_sigma = (0.4, 1.0)

        B_sku = N_ITEMS - A_sku - C_sku
        B_k_sigma = (0.2, 0.4)

        groups_ABC = [(A_sku, A_k_sigma), (B_sku, B_k_sigma), (C_sku, C_k_sigma)]

        for group in groups_ABC:

            n_items = group[0]
            k_sigma = group[1]
            articles = generate_unique_strings (LEN_NAME_ITEM, n_items)

            for article in articles:

                mu_day = uniform  ( float (MIN_AVG_SALE), float (MAX_AVG_SALE) )
                sigma_day = mu_day * uniform (k_sigma[0], k_sigma[1])

                corr_k = self._calc_corr_k (mu_day, sigma_day, SIZE_CORR)

                ost = int(mu_day * N_DAYS_IN_PER) + 1  #random.randint(0, MAX_OST)
                transit = [{'qty': 0, 'left': 0} for _ in range(MAX_TIME_TRANSIT)] # если срок поставки большой то в пути может быть неск партий

                sales_per = [] # накопитель продаж для группировки за период (напр 7 дней)
                sales_predict = [ self._gen_sales_per (mu_day, sigma_day, corr_k, N_DAYS_IN_PER) for _ in range(N_PER_PREDICT) ]

                self.item_matrix.append   ({  'article': article,
                                                'mu': mu_day,
                                                'sigma': sigma_day,
                                                'corr_k': corr_k,
                                                'ost': ost,
                                                'transit': transit,
                                                'sales_per': sales_per,
                                                'sales_predict': sales_predict
                                            })


    def one_day (self):
        # моделирование одного торгового дня
        # по всем items

        for item in self.item_matrix:

            # спрос
            spros = round (item['corr_k'] * max(0, normalvariate(item['mu'], item['sigma'])))

            if item['ost'] < spros:
                self.rew_neud_sum += spros - item['ost']
                self.rew_neud_num += 1

            # sale
            sale = min (spros, item['ost'])
            self.rew_sales += sale
            #self.rew_count_sales += 1
            item['sales_per'].append(sale) # добавление в накопитель ежедн продаж
            if len(item['sales_per']) == N_DAYS_IN_PER: # если накопитель полон. то освобождаем его
                item['sales_predict'].pop(0)
                item['sales_predict'].append(sum(item['sales_per']))
                item['sales_per'] = []

            item['ost'] -= sale

            # приезд транзита
            for tr in item['transit']:
                if tr['left'] == 0:
                    continue
                tr['left'] -= 1
                if tr['left'] == 0:
                    # партия товара приехала, оприходуем
                    item['ost'] += tr['qty']
                    tr['qty'] = 0

            self.rew_stock += item['ost']


    def _make_inp_1_item (self, item) -> list[int]:
        # создание входного вектора [ost, transit[qty,left], sales_predict[sale], TIME_TRANSIT]
        input = []
        input.append (item['ost'])
        for d in item['transit']:
            input.append(d['qty'])
            input.append(d['left'])
        for s in item['sales_predict']:
            input.append (s)
        input.append (self.time_tr)

        return input


    def len_inp_1_item (self, item):
        # расчет длины input для 1 item
        #check
        assert len(item['transit']) == MAX_TIME_TRANSIT
        assert len(item['sales_predict']) == N_PER_PREDICT

        return len(self._make_inp_1_item(item))
    

    def make_batch_inp (self, size_batch, iter) -> list[list[int]]:
        # size_batch: размер батча (сколько item обрабатывается за раз)
        # i_start_batch: нач индекс батча в item_matrix
        batch = []
        a = size_batch * iter
        b = a + size_batch
        for item in self.item_matrix[a : b]:
            batch.append(self._make_inp_1_item(item))

        return batch


    def put_zakaz_batch (self, zakaz: list[int], size_batch, iter):
            # заявка на пополнение нескольких артикулов (size_batch)
            # iter: номер батча в массиве
            for i in range(size_batch):
                qty = zakaz[i]
                
                # check
                assert qty >= 0, "ERROR: zakaz[i] < 0"
                
                if qty == 0:
                    continue
                ind = iter * size_batch + i
                item = self.item_matrix[ind]
                for tr in item['transit']:
                    if tr['left'] == 0:
                        tr['qty'] = qty
                        tr['left'] = self.time_tr
                        
                        # for check
                        qty = -1
                        break
                
                assert qty == -1, 'ERROR: qty != -1: # заказ не размещен в транзит'


    def __str__(self):
        for val in self.item_matrix:
            print (f'art={val['article']}, mu={val['mu']:.2f}, sigma={val['sigma']:.2f}, corr_k={val['corr_k']:.2f}, \
                   ost={val['ost']}, transit={val['transit']}, sales={sum([*val['sales_per']]) / (N_PER_PREDICT * N_DAYS_IN_PER):.2f}')
            #print (f'art={art}, ost={val['ost']}, transit={val['transit']}, sales={[*val['sales']]}')
        return ''
        


        
if __name__ == '__main__':

    SP = SalePoint (3)