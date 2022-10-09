import numpy as np
from scipy import stats
from scipy.stats import expon
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import arviz as az
import random
import statistics
#x = stats.expon.rvs(0, 1, 10000)
#lambda1=4 hrs^-1 => p1=1/4=0.25
p_lambda_1 = 0.25

#lamda2=6 hrs^-1 => p2=1/6=0.16
p_lambda_2 = 0.16

X = []

#generati 10.000 de valori pentru X
for n in range(1, 10_000):
    x = random.randint(1, 100)
    if x < 40:
        # mecanicul 1
        x_mecanic1 = stats.expon.rvs(0, p_lambda_1, 1)
        X.append(x_mecanic1[0])
    else:
        # mecanicul 2
        x_mecanic2 = stats.expon.rvs(0, p_lambda_2, 1)
        X.append(x_mecanic2[0])

az.plot_posterior({'x': X})
plt.show() #afisarea graficului densitatii distributiei lui X
# media ca numar
print(statistics.mean(X))
# deviatia standard
print(statistics.stdev(X))
