import matplotlib.pyplot as plt
import csv
import statistics
import arviz as az
import numpy as np
import pymc3 as pm

if __name__ == '__main__':
    ppvt = []
    moms_ages = []
    educ_cat = []

    np.random.seed(1)

    with open('data.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        
        for row in plots:
            ppvt.append(int(row[1]))
            educ_cat.append(int(row[2]))
            moms_ages.append(int(row[3]))
    
    plt.scatter(moms_ages, ppvt)
    plt.xlabel('educ_cat')
    plt.ylabel('ppvt')
    plt.title('Mom age')
    #plt.show()

    #creare modelului
    csv_model = pm.Model()
    with csv_model:     
        ppvt_sd = np.std(ppvt)
        ages_sd = np.std(educ_cat)
        alfa = pm.Normal('alfa',mu=0,sd=10*(ppvt_sd))
        beta = pm.Normal("beta",mu=0,sd=1*ppvt_sd/ages_sd)
        epsilon = pm.HalfCauchy("epsilon",5)
        miu = pm.Deterministic('miu', alfa + beta * educ_cat)
        ppvt_pred = pm.Normal('ppvt_pred', mu=miu, sd=epsilon, observed=ppvt)

    idata_g = pm.sample(2000, tune=2300, return_inferencedata=True,model=csv_model)
    #az.plot_trace(idata_g, var_names=['alfa', 'beta', 'epsilon'],show=True)
 

    #INCERCARE EX3
    alpha_m = alfa.mean()
    beta_m = beta.mean()
    ppc = pm.sample_posterior_predictive(idata_g, samples=400, model=csv_model)
    np.asarray(educ_cat)
    az.plot_trace(np.asarray(educ_cat), np.asarray(ppvt),show=True)
    az.plot_trace(np.asarray(educ_cat), alpha_m + beta_m * np.asarray(educ_cat),show=True)
    az.plot_hdi(np.asarray(educ_cat), ppc['ppvt_pred'], hdi_prob=0.5, color='grappvt', smooth=False,show=True)
    az.plot_hdi(np.asarray(educ_cat), ppc['ppvt_pred'], color='gray', smooth=False,show=True)
   # plt.xlabel('moms_ages')
   # plt.ylabel('ppvt', rotation=0)
