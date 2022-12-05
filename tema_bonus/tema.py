import numpy as np
import pymc3 as pm

if _name_ == '_main_':
    #Ex1
    # numarul de statii de gatit = 3
    # numarul de case = 2
    nr_statii_gatit = 3
    nr_case = 2
    #Ex2
    # numarul de mese = 5
    nr_mese = 5

    timp_gatire = 0
    timp_comandare = 0
    timp_mancat = 0
    counter = 0
    counter_tables = 0
   
    #folosim distributie Poisson
    trafic = np.random.poisson(20, 100)

    for i in range(0,len(trafic)):
        my_model = pm.Model()

        with my_model:
            comanda = pm.Normal('N', sigma=0.5, mu=1)
            gatit = pm.Exponential('G', 1/2)
            mancat = pm.Normal('M', mu=10, sigma=2)

        trace = pm.sample(trafic[i], chains=1, model=my_model)
        
        result = {
            'comanda': trace['N'].tolist(),
            'gatit': trace['G'].tolist(),
            'mancat': trace['M'].tolist()
        }

        for elem in result['comanda']:
            timp_comandare += elem

        for elem in result['gatit']:
            timp_gatire += elem

        for elem in result['mancat']:
            timp_mancat += elem

        timp_gatire /= nr_statii_gatit
        timp_comandare /= nr_case
        timp_mancat /= nr_mese

        if timp_gatire + timp_comandare <= 60:
            counter += 1

        if timp_mancat <= 60:
            counter_tables += 1

    print(counter/100)
    print(counter_tables/100)
