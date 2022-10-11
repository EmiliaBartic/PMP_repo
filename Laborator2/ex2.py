import numpy as np
from scipy import stats
from scipy.stats import gamma
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import arviz as az
import random
#import statistics

# avem de-a face cu 4 servere

lambda_milisec = 4
p_server1 = 0.25  # de la 0 la 24
p_server2 = 0.25  # de la 25 la 50
p_server3 = 0.30  # de la 51 la 81
p_server4 = 1/3  # 1-0.25-0.25-0.30 ; 82-100

nr = 0

X = []

for i in range(1000):
    latenta = stats.expon.rvs(0, 1/4, 1)
    x = np.random.randint(0, 100)
    if x < 25:
        # server 1
        # gamma(4,3)
        timp_server = gamma.rvs(4, 0, 1/3 , 1)[0]
        timp_servire = timp_server + latenta
        X.append(timp_servire)
        
    elif x>=25 and x < 50:
        # server 2
        # gamma(4,2)
        timp_server = gamma.rvs(4, 0, 1/2 , 1)[0]
        timp_servire = timp_server + latenta
        X.append(timp_servire)
       
    elif x>=50 and x < 80:
        # server 3
        # gamma(5,2)
        timp_server = gamma.rvs(4, 0, 1/2, 1)[0]
        timp_servire = timp_server + latenta
        X.append(timp_servire)
       
    else:
        # server 4
        # gamma(5,3)
        timp_server = gamma.rvs(4, 0, 1/3, 1)[0]
        timp_servire = timp_server + latenta
        X.append(timp_servire)
       
nr = 0
for x in X:
    if x > 3:
        nr = nr + 1 


# print('Probabilitatea ca timpul necesar servirii unui client sa fie mai mare decat 3 milisecunde este:')
print(nr/1000)
az.plot_posterior({'x': X})
plt.show()  # afisarea graficului densitatii distributiei lui X
