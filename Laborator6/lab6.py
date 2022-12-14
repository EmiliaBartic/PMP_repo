import matplotlib.pyplot as plt
import csv
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
    plt.xlabel('moms_ages')
    plt.ylabel('ppvt')
    plt.title('Mom age')
    #plt.show()

    #creare modelului
    csv_model = pm.Model()
    with csv_model:
        #priors for unknown model parameters
        #ppvt_sd / ages_sd  
        ppvt_sd = np.std(moms_ages)
        ages_sd = np.std(ppvt)
        alfa = pm.Normal('alfa',mu=0,sd=10*ppvt_sd)
        beta = pm.Normal("beta",mu=0,sd=1*ppvt_sd/ages_sd)
        epsilon = pm.HalfCauchy("epsilon",5)
        miu = pm.Deterministic('miu', alfa + beta * ppvt)
        ppvt_pred = pm.Normal('ppvt_pred', mu=miu, sd=epsilon, observed=moms_ages)

    idata_g = pm.sample(2000, tune=2000, return_inferencedata=True,model=csv_model)
    az.plot_trace(idata_g, var_names=['alfa', 'beta', 'epsilon'])

#INCERCARE EX3
#alpha_m = alfa.mean().item()
# beta_m = beta.mean().item()
# ppc = pm.sample_posterior_predictive(idata_g, samples=400, model=csv_model)
# plt.plot(moms_ages, ppvt, 'b.')
# plt.plot(moms_ages, alpha_m + beta_m * moms_ages, c='k',
# label=f'ppvt = {alpha_m:.2f} + {beta_m:.2f} * moms_ages')
# az.plot_hdi(moms_ages, ppc['ppvt_pred'], hdi_prob=0.5, color='grappvt', smooth=False)
# az.plot_hdi(moms_ages, ppc['ppvt_pred'], color='grappvt', smooth=False)
# plt.xlabel('moms_ages')
# plt.ylabel('ppvt', rotation=0)
