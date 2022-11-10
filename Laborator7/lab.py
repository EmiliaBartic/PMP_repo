import math
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

var = pd.read_csv('Prices.csv')
input_prices = var['Price'].values
input_speed = var['Speed'].values
input_hdd = var['HardDrive'].values
#input_ram = var['Ram'].values
#premium = var['Premium'].values

#EX1 - definim modelul
with pm.Model() as modeel:
    alfa2 = pm.Normal('alfa_tmp', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=1)
    beta2 = pm.Normal('beta2', mu=0, sd=1)
    epsilon = pm.HalfCauchy('ε', 5)

    new_mu = alfa2 + pm.math.dot(input_speed, beta1) + pm.math.dot(math.log(input_hdd), beta2)
    alfa = pm.Deterministic('α', alfa2 - pm.math.dot(np.mean(input_speed), beta1) + pm.math.dot(np.mean(math.log(input_hdd)), beta2))
    y_pred = pm.Normal('y_pred', mu=new_mu, sd=epsilon, observed=input_prices)
    idata_mlr = pm.sample(2000, return_inferencedata=True)

az.plot_trace(idata_mlr, var_names=['alfa', 'beta1', 'beta2', 'epsilon'])


# checking_model = modeel.check_model()
# if checking_model == True:
#     print('Model ok')
# else:
#     print('Model not ok')
