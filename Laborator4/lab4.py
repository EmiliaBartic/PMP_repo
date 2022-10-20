import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()
nr_clienti = pm.Model()

''''
with client_number:
    number=pm.Poisson('R',20)
    trace= pm.sample(1,chains=1)

dictionary= {
    'number': trace['R'].tolist()
}

'''
with nr_clienti:
    number = pm.Poisson('NRC', 20)
    trace = pm.sample(1, chains = 1)

dict= {
    'nr': trace['NRC'].tolist()
}
'''
dictionary = {
    'nr_clienti': trace['N'].tolist(),
    'timp_plata': trace['T'].tolist()
}

'''
plata = list()
preparare = list()

with model:

    nr_clienti = pm.Poisson('N', 20)

    for i in range(0, dict['nr'][0]):        
        pay_time = pm.Normal('T' + str(i), 1, 0.5)
        timp_preparare = pm.Exponential('S' + str(i), 3)

        plata.append(pay_time)
        preparare.append(timp_preparare)
