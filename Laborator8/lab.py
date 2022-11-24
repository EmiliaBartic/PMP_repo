#!pip install pgpy
#!pip install pandas
#am rulat in google colab
!pip install pymc3
import math
import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az

def main():
    data = pd.read_csv('./Admission.csv')
    df=data
    y_1 = pd.Categorical(df['Admission']).codes
    
    x_n = ['GRE', 'GPA']
    first_axis_x1 = df[x_n].values

    with pm.Model() as model_1:
        
        α = pm.Normal('α', mu=0, sd=10) #alfa
        β = pm.Normal('β', mu=0, sd=2, shape=len(x_n)) #beta
        μ = α + pm.math.dot(first_axis_x1, β)

        θ = pm.Deterministic('θ', 1 / (1 + pm.math.exp(-μ))) 
        bd = pm.Deterministic('bd', -α/β[1] - β[0]/β[1] * first_axis_x1[:,0])
        yl = pm.Bernoulli('yl', p=θ, observed=y_1)
        idata_1 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)

    #datele si frontiera de decizie
    sort_idx = np.argsort(first_axis_x1[:,0])
    bd = idata_1.posterior['bd'].mean(("chain", "draw"))[sort_idx]
    plt.scatter(first_axis_x1[:,0], first_axis_x1[:,1], c=[f'C{x}' for x in y_1])
    plt.plot(first_axis_x1[:,0][sort_idx], bd, color='k')
    #reprezentare grafica 
    az.plot_hdi(first_axis_x1[:,0], idata_1.posterior['bd'], color='k')
    plt.xlabel(x_n[0])
    plt.ylabel(x_n[1])

if __name__ == '__main__':
    main()
