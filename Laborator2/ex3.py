import numpy as np
from scipy import stats
from scipy.stats import gamma

import matplotlib.pyplot as plt
import arviz as az
import random

# moneda nemasluita
p_m1 = 0.5
# moneda masluita
p_m2_stema = 0.3
p_m2_ban = 0.7

SS = []
SB = []
BS = []
BB = []
# sa se genereze 100 de rezultate independente
for repetare in range(1, 100):
    # aruncarea de 10 ori a doua monezi
    ss = 0
    sb = 0
    bs = 0
    bb = 0
 
    for n in range(1, 10):
        # do something

        m1 = random.random()  # nemasluita
        m2 = random.random()  # masluita

        if m1 <= 0.5:
            # prima moneda este stema
            moneda1 = 'stema'
        elif m1 > 0.5:
            # a doua moneda este ban
            moneda1 = 'ban'
        if m2 <= 0.3:
            moneda2 = 'stema'
        else:
            moneda2 = 'ban'
        #print(moneda1, moneda2)
        #print(m1, m2)
        if moneda1 == 'stema' and moneda2 == 'stema':
            ss = ss + 1
        elif moneda1 == 'stema' and moneda2 == 'ban':
            sb = sb + 1
        elif moneda1 == 'ban' and moneda2 == 'stema':
            bs = bs + 1
        else:
            bb = bb + 1
        SS.append(ss)
        SB.append(sb)
        BS.append(bs)
        BB.append(bb)

# determinare grafica?
az.plot_posterior({'SS': SS})
plt.show()
az.plot_posterior({'SB': SB})
plt.show()
az.plot_posterior({'BS': BS})
plt.show()
az.plot_posterior({'BB': BB})
plt.show()

  
