import numpy as np
from scipy import stats
from scipy.stats import expon
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import arviz as az
import random
import statistics
#x = stats.expon.rvs(0, 1, 10000)

p1 = 0.25
p2 = 0.16
X = []
for n in range(1, 10000):
    x = random.randint(1, 100)
    if x < 40:
        # mecanicul 1
        x_m1 = stats.expon.rvs(0, p1, 1)
        X.append(x_m1[0])
    else:
        # mecanicul 2
        x_m2 = stats.expon.rvs(0, p2, 1)
        X.append(x_m2[0])

az.plot_posterior({'x': X})
plt.show()
# media ca numar
print(statistics.mean(X))
# deviatia standard
print(statistics.stdev(X))
