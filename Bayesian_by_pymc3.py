# Let's say 90% of the people with mutated x gene develop y cancer
# what is the probability that someone with mutated x gene develop cancer y given a specific observation?
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from matplotlib import MatplotlibDeprecationWarning

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning)

import pymc3 as pm
groups = ['healthy', 'cancer']
# alphas = prior belief (1 in 10 is healthy); data = observation (1 healthy and 1 cancerous)
alphas = np.array([1, 9])
data = np.array([1,1])

with pm.Model() as model:
    parameters = pm.Dirichlet('parameters', a= alphas, shape = 2)
    observation = pm.Multinomial('observation', n = 200, p = parameters, shape = 2, observed = data)
    
with model:
    trace = pm.sample(draws = 1000, chains = 2, tune = 500, discard_tuned_samples = True)

pm.plot_posterior(trace, varnames = ['parameters'])
plt.show()   
#The observed sample size is rather small. The posterior probility remains similar to the prior belief
# see the probabilities from all the draws
pm.traceplot(trace)
plt.show() 
#The probability of being healthy fluctuates between 0 and 0.5 while being cancerous flucturates between 0.5 and 1.

summary = pm.summary(trace)
np.around to round up the numbers to the 2nd decimal places
prob_df = pd.DataFrame(trace.parameters)
sns.distplot(prob_df[0], color = 'blue', bins = 30, label = 'healty')
sns.distplot(prob_df[1], color = 'red', bins = 30, label = 'cancer')
plt.xlabel('probability')
plt.ylabel('count')
plt.title('Posterior probabilities with # of draws')
plt.legend()
plt.show()

print('The posterior probability of a person having the cancer y if x gene is mutated is:', np.around(summary['mean'][1], decimals = 2), 'with 95% CI between', 
    np.around(summary['hpd_2.5'][1], decimals = 2), 
    'and' ,np.around(summary['hpd_97.5'][1], decimals = 2))
# If the observation is changed to have a lot more weight than before
new_data = np.array([1000,1000])
with pm.Model() as new_model:
    parameters = pm.Dirichlet('parameters', a= alphas, shape = 2)
    observation = pm.Multinomial('observation', n = 2000, p = parameters, shape = 2, observed = new_data)
    
with new_model:
    new_trace = pm.sample(draws = 1000, chains = 2, tune = 500, discard_tuned_samples = True)
 
pm.plot_posterior(new_trace, varnames = ['parameters'])
plt.show() 
# with new data where the sample number is significantly more, the new probabilities are heavily influenced by new data
pm.traceplot(trace)
plt.show() 
# the probabilities of being healthy and cancerous is about half and half

new_summary = pm.summary(new_trace)
print('The posterior probability of a person having the cancer y if x gene is mutated is:', np.around(new_summary['mean'][1], decimals = 2), 'with 95% CI between', 
    np.around(new_summary['hpd_2.5'][1], decimals = 2), 
    'and' ,np.around(new_summary['hpd_97.5'][1], decimals = 2))

# Last, let's try a strong prior belief that has 10,000 data points
new_alpha = np.array([1000,9000])
new_data = np.array([90,10])
with pm.Model() as new_model:
    parameters = pm.Dirichlet('parameters', a= new_alpha, shape = 2)
    observation = pm.Multinomial('observation', n = 100, p = parameters, shape = 2, observed = new_data)
    
with new_model:
    new_trace = pm.sample(draws = 1000, chains = 2, tune = 500, discard_tuned_samples = True)
 
pm.plot_posterior(new_trace, varnames = ['parameters'])
plt.show()
pm.traceplot(trace)
plt.show()
# it is clear to see that the probabilities are very similar to the prior belief
final_summary = pm.summary(new_trace)
print('The posterior probability of a person having the cancer y if x gene is mutated is:', np.around(final_summary['mean'][1], decimals = 2), 'with 95% CI between', 
   np.around(final_summary['hpd_2.5'][1], decimals = 2), 
   'and' ,np.around(final_summary['hpd_97.5'][1], decimals = 2))
