
# CONFIG

# salepoint
N_PER_PREDICT = 8 # число периодов (недель) продаж по которым будем делать прогноз продаж
N_DAYS_IN_PER = 7 # число дней в периоде
LEN_NAME_ITEM = 5 # длина имени артикула
SIZE_CORR = 10000 # кол циклов в расчете поправочного коэф для сигмы (чем больше тем точнее)
MIN_AVG_SALE = 1
MAX_AVG_SALE = 10 
MAX_TIME_TRANSIT = 10 # размер массива под пополнения == макс числу дней транзита

# iters
N_ITEMS = 100 # кол артикулов
TIME_WORK = 100 # сколько циклов (дней) прогоняем агентов

# net
SIZE_BATCH = 10 # кол артикулов обрабатываемых за один просчет НН
INP_SIZE = None # None: размер входного слоя зависит от размера входа, будет определен после создания SP
SIZES_NET = [INP_SIZE, 64, 64, 1] # 

# populations
SIZE_TOUR = 3 # совет Gemma4 # кол сравниваемых юнитов в туре
N_TOUR  = 10 # кол туров в турнире
N_POPUL = 2 + SIZE_TOUR * N_TOUR # = 32 # 2: elite + worst, 10: count tour selection

# cross
#KOEFF_CROSS = [0.5, 0.5] # базовый случай 
KOEFF_CROSS_EL_WO = [0.7, 0.3] # коэф скрещивания элитного с худшим

# mutation
PROB_MUT = 0.05 # вероятность мутации одного веса 
MIN_SIGMA_MUT = 0.01 # use in adaptive mutation
MAX_SIGMA_MUT = 0.5  # use in adaptive mutation
MIN_WEIGHT = -100.0 # min weight of net
MAX_WEIGHT = 100.0  # max weight of net