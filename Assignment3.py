
# coding: utf-8

# # Assignment 3: Confidence Intervals & The Bootstrap
# 
# ### Instructions
# 
# This assignment is much like the last.
# 
# * This assignment includes some tests to help you make sure that your implementation is correct.  When you see a cell with `assert` commands, these are tests.
# 
# * Once you have completed the assignment, delete the cells which have these `assert` commands.  You will not need them.
# 
# * When you are done and have answered all the questions, convert this notebook to a .py file using `File > Download as > Python (.py)`.  Name your submission `assignment3.py` and submit it to OWL.
# 
# Failure to comply may resilt in you not earning full marks for your assignment.  We want you to earn full marks!  Please follow these instructions carefully.

# # Question 1
# 
# ### Part A
# 
# Recall from theclture that a $199(1-\alpha)\%$ confidence interval for the mean is 
# 
# $$ \bar{x} \pm  t_{1-\alpha/2, n-1} \dfrac{\hat{\sigma}}{\sqrt{n}} $$
# 
# Where $ t_{1-\alpha/2, n-1}$ is the appropiorate quantile of a Student's t distribution with $n-1$ degrees of freedom.  When $\alpha = 0.05$ and when $n$ is big enough, $ t_{1-\alpha/2, n-1} \approx 1.96$.  
# 
# Write a function called `confidence_interval` which takes as it's argument an array of data called `data` and returns two things:
# 
# * An estimated mean of `data`, and 
# 
# * The lower and upper bounds of the 95% confidence interval for the mean of `data`.  Ensure these are returned in a numpy array of shape (2,)

# In[10]:


#It's dangerous to go alone.  Take these
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
import scipy
def confidence_interval(data):
    '''
    Function to compute confidence interval (ci).
    
    Inputs:
    data - ndarray.  Data to be used to compute the interval
    
    Outputs:
    estimated_mean - float.  The mean of the data
    bounds - array. An array of length 2 where bounds[0] is the lower bound and bounds[1] is the upper bound for the ci.
    '''
    ### BEGIN SOLUTION
    estimated_mean  = data.mean()
    stderr = data.std()/np.sqrt(len(data))
    my_df = len(data) - 1
    critval = scipy.stats.t.ppf(0.975, df=my_df)
    bounds = [estimated_mean - critval*stderr, 
        estimated_mean + critval*stderr]
    ### END SOLUTION
    
    return estimated_mean, np.array(bounds)
confidence_interval(np.array([-1,0,1]))


# ### Tests
# 
# Tests are to make sure you've implemented the solution correctly.  If these tests pass without any `AsserstionError`'s, then you can be confident that you've implemented the solution as expected.
# 
# Once you're happy with your implementation, delete the cell below.

# ### Part B
# 
# The "95% confidence interval" is named so because the long term relative frequency of these estimtors containing the true estimand is 95%.  That is to say **if I construct 95% confidence intervals for the sample mean again and again from the same data generating mechanism, 95% of the intervals I construct will contain the population mean**.
# 
# Write a function called `ci_simulation` that runs some simulations to show this is the case.  From a standard normal distirbution, sample 25 observations and construct a confidence interval.  Do this 20 times and plot the intervals using `matplotlib.pyplot.errorbar`. Save your plot under the name `ci_simulation.png`.  Color the bar red if the confidence interval does not caputre the true mean and blue if it does.  If you are unfamilliar with `matplotlib.pyplot.errorbar`, I highly suggest reading Matplotlib's excellent documentation.

# In[12]:


def ci_simulation():
    # Reporducibility.  Do not change this!
    np.random.seed(3)
    
    # Create the figure.
    fig, ax = plt.subplots(dpi = 120)

    # If the interval crosses this line, it should be blue, else red.
    ax.axhline(0, color = 'k')
    CI_Out = np.zeros(20)
    s_mean_out = np.zeros(20)
    s_stderr_out = []
    norm_ci_out = []
    list_of_CI = []
    ### BEGIN SOLUTION
    for i in range(20):
        s = np.random.standard_normal(25)
        s_mean = s.mean()
        s_mean_out[i] = s_mean
        s_stderr = s.std()/np.sqrt(len(s))
        critval = 1.96
        s_stderr_out.append(critval*s_stderr)
        list_of_CI.append([s_mean - critval*s_stderr, s_mean + critval*s_stderr])
        norm_ci_out.append([np.around(s_mean - critval*s_stderr,decimals=2, out=None), np.around(s_mean + critval*s_stderr, decimals=2, out=None)])
    #print(f"list_of_CI: {list_of_CI}")
    for j in range(20):
        if norm_ci_out[j][0]<= 0.0 <= norm_ci_out[j][1]:
            plt.plot(np.array([j]),np.array(s_mean_out[j]),'bo')
            plt.errorbar(np.array([j]),s_mean_out[j],yerr=np.array(s_stderr_out[j]),linestyle='', ecolor = 'blue')
        else:
            plt.plot(np.array([j]),np.array(s_mean_out[j]),'ro')
            plt.errorbar(np.array([j]),s_mean_out[j],yerr=np.array(s_stderr_out[j]),linestyle='', ecolor = 'red')
    plt.savefig('ci_simulation.png')        
    plt.show()
ci_simulation()


# ### Part C
# 
# If you haven't changed the random seed from 3 and if you implemented the solution correctly, you should two red intervals.
# 
# Answer the following below in 1-2 sentences:
# 
# 1) How many red intervals did we expect to see?  What is your justifiation for this?
# Solution: Based on the bold sentense above, 95% of the intervals we construct will contain the population mean. We have 20 intervals and 95 percent of them ((95/100)*20 = 19) is 19 which means 19 out of 20 will contain the mean
# 2) If there is a discrepency between the number of observed and expected red intervals, what explains this difference?
# Solution:maybe the number of confidence interval we are looking at is not enough. If we increase sample mean that might help
# 

# ### Part D
# 
# How many samples would we need in order to ensure that our constructed confidence interval is approximately 0.1 units long? 
# 
# Write a function called `num_propper_length` which takes as its only argument an integer `n`.  `num_propper_length` should simulate 1000 datasets of length `n`, compute confidence intervals for those datasets, compute the lengths of those intervals, and then returns the number of intervals which are no longer than 0.1 units.
# 
# Determine how many samples you need (that is, compute `n`).  Set this as your default argument for `n` in `num_propper_length`.

# In[17]:


def num_propper_length(n=1600):
    '''
    Function to simulate how many out of 1000 confidence intervals
   would be no longer than 0.1 units long if
    we sampled n observations from a standard normal.
    
    Inputs:
        n - int.  Number of draws to make from the standard normal
        
    Outputs:
        num_long_enough - integer.  Number of constructed intervals which are no longer than 0.1.
    '''
    # For reproducibility.  Don't change this!
    np.random.seed(0)
    
    ### BEGIN SOLUTION
    list_of_CI = []
    counter = 0
    s_stderr_out_out = []
    for i in range(1000):
        s = np.random.standard_normal(n)
        s_mean = s.mean()
        s_dev = s.std()
        s_stderr = s_dev/np.sqrt(len(s))
        my_df = len(s) - 1
        critval = scipy.stats.t.ppf(0.975, df=my_df)
        list_of_CI.append([s_mean - critval*s_stderr, s_mean + critval*s_stderr])
        s_stderr_out_out.append(2*critval*s_stderr)
    for j in range(1000):  
        if  s_stderr_out_out[j] < 0.1:
            counter += 1
        num_long_enough = counter
    return num_long_enough
num_propper_length(1600)


# ### Tests
# 
# Tests are to make sure you've implemented the solution correctly.  If these tests pass without any `AsserstionError`'s, then you can be confident that you've implemented the solution as expected.
# 
# Once you're happy with your implementation, delete the cell below.

# In[14]:





# ### Part E
# If you chose the right `n`, you should find that 891 (or approximately 89%) of your intervals are longer than 0.1.  
# 
# Why is this?  Answer below in 1-2 sentences.
# 

# ---

# ## Question 2
# 
# ### Part A
# The dataset `hockey_stats.csv` contains information about information about hockey draftees.  We'll use this data to investigate the relationship between height and age on weight.  Load it into python using pandas.
# 
# Load in the `hockey_draftees_train.csv` data into pandas.  Fit a linear model of weight (`wt`) explained by height (`ht`) and age(`age`).  Call your fitted model `model`.

# In[25]:


### BEGIN SOLUTION
data = pd.read_csv("hockey_draftees_train.csv") 
data.head()
#model1 = smf.ols("wt~ht + age", data = data).fit()
#print(model1.summary())
X = data[['ht','age']]
y = data['wt']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
### END SOLUTION


# ### Part B
# 
# Print out the R-squared for this model
# 

# In[26]:




print('Training rsquared is ',model.rsquared)



# ### Part C
# 
# Now, let's see how well our model performs out of sample.  Load in the `hockey_draftees_test.csv` file into a dataframe.  Use your `model` to make predictions, and print the r-squared on the out of sample (oos) data.

# In[36]:


### BEGIN SOLUTION
data_test = pd.read_csv("hockey_draftees_test.csv") 
data_test.head()
X = data_test[['ht','age']]
y = data_test['wt']
X1 = sm.add_constant(X)
ypred = model.predict(X1)
y_hat = np.array(ypred)
y_observed  = np.array(y)
R2 = 1-(np.sum((y_observed-y_hat)**2))/(np.sum((y_observed-y_observed.mean())**2))
print('Out of sample rsquared is ', R2)

### END SOLUTION
#it is interesing because when I run the code through online version of jupyter the output is exactly 
#the same as yours
#print('Out of sample rsquared is ', rsquared_oos)
#Out of sample rsquared is  0.3313198566514667


# ### Part D
# 
# A point estimate of the rsquared is nice, but what we really want is uncertainty estimates.  For that, we need a confidence interval.  To estimate how uncertain we are in our out of sample r-squared, let's use the bootstrap.
# 
# Write a function called `bootstrap` which takes three arguments:
# 
# * `data`, which is a dataframe, and 
# * `model` which is an statsmodel ols model. `data` should look the the data `model` was trained on so that we can use `model` to make predictions on `data`.
# * `numboot` which is an integer denoting how many bootstrap replications to perform.
# 
# Write `bootstrap` to perform bootstrap resampling for the out of sample r-squared.  You can use `pd.DataFrame.sample` with `replace = True` to perform the resampling.
# 
# `bootstrap` should return a numpy array of bootstraped rsquared values.
# 
# 

# In[37]:


def bootstrap(data, model, numboot):
    '''
    Function to bootstrap the r-squared for a linear model
    
    Inputs:
        data - dataframe.  Data on which model can predict.
        model - statsmodel ols.  Linear model of weight explained by height and age.  Can predict on data.
        numboot - int.  Number of bootstrap replications.
    
    Outputs:
        bootstrapped_rsquared - array.  An array of size (numboot, ) which contains oos bootstrapped rsquared values.
    
    '''
    ### BEGIN SOLUTION
    model_in = model
    data_in  = data
    bootstrapped_rsquared = np.zeros(numboot)
    for i in range(numboot):
        d = data_in.sample(n=1121,replace=True)
        X = d[['ht','age']]
        y = d['wt']
        X1 = sm.add_constant(X)
        #print(np.array(X1))
        ypred = model_in.predict(X1)
        y_hat = np.array(ypred)
        y_observed  = np.array(y)
        R2 = 1-(np.sum((y_observed-y_hat)**2))/(np.sum((y_observed-y_observed.mean())**2))
        bootstrapped_rsquared[i] = R2
    ### END SOLUTION

    return bootstrapped_rsquared
 
boot_rsquared = bootstrap(pd.read_csv("hockey_draftees_test.csv"),model,10000)
print(boot_rsquared)


# ### Part E
# 
# Use your `bootstrap` function to plot 10,000 bootstrap replicates as a histogram.

# In[31]:


### BEGIN SOLUTION
plt.hist(boot_rsquared)
plt.show()
### END SOLUTION


# ### Part F
# 
# Use your bootstrap replicates to estimates to obtain a bootstrapped 95% confidence interval.  Call the upper confidence bound `ci_upper` and the lower confidence bound `ci_lower`.

# In[ ]:


loss_boot_ci = np.quantile(boot_rsquared, [0.025, 0.975])
ci_lower = loss_boot_ci[0]
ci_upper = loss_boot_ci[1]
print('My confidence interval is between', ci_lower, ' and ', ci_upper)

