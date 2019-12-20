#%%
import numpy as np
import pandas as pd

#%%
import pymc3 as pm
# I can't believe the blood sweat and tears 
# that had to go into making this run

# %%
from random import choices
from scipy import stats
import altair as alt


# %% Chapter 2 - Garden of Forking Data
# Suppose there's a bag and it contains 4 marbles
# Marbles only come in blue and white
# There are 5 conjectures:
# 4B, 3B1W, 2B2W, 1B3W, 4W

# We draw with replacement and see
# Blue, White, Blue

#%%
class Conjecture():
    def __init__(self,white,blue):
        self.white=white
        self.blue=blue
    def __str__(self):
        return f'{self.white}W,{self.blue}B'

# %%
c1=Conjecture(4,0)
c2=Conjecture(3,1)
c3=Conjecture(2,2)
c4=Conjecture(1,3)
c5=Conjecture(0,4)
# %%
def num_of_ways_to_get_BWB(c):
    # Blue
    ans=c.blue
    # White
    ans*=c.white
    # Blue
    ans*=c.blue
    return ans


# %%
ways=[]
for c in [c1,c2,c3,c4,c5]:
    ways.append(num_of_ways_to_get_BWB(c))

# %%
def plausibility(ways):
    return [x/sum(ways) for x in ways]

# %%
plausibility(ways)

# %% How to generate a random nums from 
# Binomial Distribution (Numpy)

n=9
p=0.5
np.random.binomial(1,p,n)
#array([1, 1, 0, 0, 1, 1, 0, 1, 0])

# %% Random nums from Binomial Distribution (Scipy)
stats.binom.rvs(1,p,size=n)
#array([0, 0, 1, 1, 0, 1, 0, 0, 0])

#%% Plot distribution

# Likelihood Formula for Binomial Distribution
# Binom(k) ~ (n choose k) * p^k * (1-p)^(n-k)

n=9
p=0.5
k=6

#%% Probability Mass Function
# This is the likelihood
# that we draw 6 (k) items (eg.water) from 
# a process that is binomially distributed
# with (0.5) prob on each toss and 9 (n) tosses

# It's also the relative num of ways
# to see data w, given p and n
# P(w | p,n)
stats.binom.pmf(k,n,p)
#0.16406250000000006

# %% 
# Grid Approximation Example
# The posterior parameter is continuous
# and can take on an infinite number of values
# But we can approximate a continuous posterior using only
# a finite grid of prior parameters

# At any particular value of the grid prior (p)
# it's simple to compute posterior
# Repeat this for all values in the grid and we can 
# approximate the exact posterior distribution

# Ultimately, grid approximation scales very poorly
# as num of parameters increase


#%% when grid is small (n=20)
# Create 101 random probabilities between 0 and 1
p_grid=np.linspace(start=0,stop=1,num=20) 

# Assume uniform prior of 1
prob_prior=np.ones(20)

# Create data by drawing from a binom distribution 
# based on the 101 random probabilities
prob_data=stats.binom.pmf(k,n,p=p_grid)

# Update the posterior
posterior = prob_data * prob_prior

# Normalize
posterior = posterior / sum(posterior) 

# %% Visualizing posterior
aux=(pd.DataFrame(posterior)
    .reset_index()
    .rename({0:'prob'},axis=1)
    )
aux['p']=aux['index']/100

grid_20=(alt.Chart(aux)
    .mark_line()
    .encode(
        x=alt.X('p',title='p'),
        y=alt.Y('prob',title='density')
    ))

#%%  when grid is bigger (n=101)
# Create 101 random probabilities between 0 and 1
p_grid=np.linspace(start=0,stop=1,num=101) 

# Assume uniform prior of 1
prob_prior=np.ones(101)

# Create data by drawing from a binom distribution 
# based on the 101 random probabilities
prob_data=stats.binom.pmf(k,n,p=p_grid)

# Update the posterior
posterior = prob_data * prob_prior

# Normalize
posterior = posterior / sum(posterior) 

# %% Visualizing posterior
aux=(pd.DataFrame(posterior)
    .reset_index()
    .rename({0:'prob'},axis=1)
    )
aux['p']=aux['index']/100

grid_101=(alt.Chart(aux)
    .mark_line()
    .encode(
        x=alt.X('p',title='p'),
        y=alt.Y('prob',title='density')
    ))

#%% observe grid approx side by side
alt.vconcat(grid_20,grid_101)


#%% Computing Posterior
# P(a and b) = P(a | b) * P(b)

# %% Sampling from our posterior
samples = (pd.DataFrame(np.random.choice(p_grid,5000,p=posterior))
    .reset_index()
    .rename({0:'prob'},axis=1)
    )
samples.head()

# %%
# The scatterplot shows a birds eye view of 
# the area plot (shifted 90 degrees)
scatterplot=(alt.Chart(samples)
    .mark_point()
    .encode(
        x=alt.X('index',title='samples'),
        y=alt.Y('prob',title='parameter p of the posterior')
    ))

areaplot=(alt.Chart(samples)
    .mark_area(
        opacity=0.3,
        interpolate='step')
    .encode(
        alt.X(
            'prob:Q',
            bin=alt.Bin(maxbins=200),
            scale=alt.Scale(domain=(0,1)),
            title='parameter p of the posterior'),
        alt.Y(
            'count()',
            stack=None,
            title='Number of records')
    ))

alt.hconcat(scatterplot,areaplot)

# %% Calculating Credible Intervals

#%% Using Numpy
(round(np.percentile(np.array(samples.prob),2.5),2),
round(np.percentile(np.array(samples.prob),97.5),2)
)

# %% Using pymc3
pm.stats.quantiles(np.array(samples.prob),qlist=[2.5,97.5])
#quantiles function depreciated?

# %% Exercise 1

