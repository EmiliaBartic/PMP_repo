import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

def initialize_dictionary() :
    result = dict()
    result = {
        'timp_comanda': trace['N'].tolist(),
        'timp_gatit': trace['G'].tolist()
    }
    return result


alfa = 2.35
counter = 0
servingTime = list()
trafic = np.random.poisson(20, 100)

for i in range(len(trafic)):
    second_model = pm.Model()

    with second_model:
        timp_comanda = pm.Normal('N', sigma=0.5, mu=1)
        timp_gatit = pm.Exponential('G', 1/alfa)

    
    trace = pm.sample(trafic[i], chains=1, model=second_model)

    dictionary = initialize_dictionary()

    dataFrame = pd.DataFrame(dictionary)

    temp = dataFrame.copy()
    filter = dataFrame['timp_comanda'] + dataFrame['timp_gatit'] < 15

    servingTime.extend(dataFrame['timp_comanda'] + dataFrame['timp_gatit'])
    dataFrame.where(filter, inplace=True)

    dataFrame = dataFrame.dropna()
   
    if (dataFrame.shape[0]/temp.shape[0]) == 1.0:
        counter += 1

print(counter/100)
print(np.mean(servingTime))
