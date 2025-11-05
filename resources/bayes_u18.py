from scipy.stats import beta, binom
from math import log, exp
from scipy.misc import logsumexp
import random
from numpy.random import choice, binomial
import sys # to print things out while the iterate function is running

###########################################
# data and grid of possible probabilities #  
###########################################
# Training data is a list of lists of counts for each condition
#                 pattern 1     pattern 2     pattern 3     pattern 4 
#                (Adj-N,Num-N) (N-Adj,N-Num) (N-Adj,Num-N) (Adj-N,N-Num)               
training_data = [[28,12,28,12],[12,28,12,28],[12,28,28,12],[28,12,12,28]]

# set up the grid
grid_granularity = 100 # granularity of grid
possible_p = []
for i in range(1,grid_granularity):
    possible_p.append(i/(grid_granularity+0.))

#######################
# Likelihood function #  
#######################                              

def U18_likelihood(data,p_AdjN,p_NumN):
    '''Calculates log likelihood of data, (where data are counts representing
    number of Adj-N out of total Adj instances and
    number of Num-N out of total Num instances)
    given point probability of Adj-N, and Num-N
    '''
    loglikelihood = [];
    loglikelihood_AdjN = binom.logpmf(data[0],data[0]+data[1],p_AdjN)
    loglikelihood_NumN = binom.logpmf(data[2],data[2]+data[3],p_NumN)
    loglikelihood = loglikelihood_AdjN + loglikelihood_NumN #summing in log domain = multiplying in non-log
    return loglikelihood                                                                                                       

##################
# Prior function #  
################## 

def U18_prior(g, a, b, p_AdjN, p_NumN):
    '''
    Calculates the log prior probability of a given p_AdjN and p_NumN given a set of parameters.
    This is a sum over the probabilities given by each mixture component.
    Parameter are: g (set of four mixture weights), a(lpha), b(eta) (Beta shape parameters)
    '''
    pattern1_component = [a,b,a,b] # higher prob for Adj-N, Num-N 	
    pattern2_component = [b,a,b,a] # higher prob for N-Adj, N-Num
    pattern3_component = [b,a,a,b] # higher prob for N-Adj, Num-N
    pattern4_component = [a,b,b,a] # higher prob for Adj-N, N-Num
    components = [pattern1_component,pattern2_component,pattern3_component,pattern4_component]
    
    logprior=[]
    for i in range(0,4): # loop over all four components
	logprior_i_Adj = beta.logpdf(p_AdjN,components[i][0],components[i][1])
	logprior_i_Num = beta.logpdf(p_NumN,components[i][2],components[i][3])
	logprior_i = logprior_i_Adj + logprior_i_Num
	logprior.append(logprior_i)
    # a+b+... in log space = log(exp(a)+exp(b)+...)
    logprior = log((g[0]*exp(logprior[0])) + (g[1]*exp(logprior[1])) + (g[2]*exp(logprior[2])) + (g[3]*exp(logprior[3])))
    return logprior

###########################################################
# Parameter values determined by fitting the testing data #
###########################################################
#                   gamma                  alpha, beta
fit_parameters = [[0.6293,0.3706,0.0001,0],16.5, 0.001]

######################
# Posterior function #  
###################### 

def U18_posterior(g,a,b,data):
    '''Calculates the log posterior probability of a set of counts 
    for all possible p_AdjN, p_NumN combinations,
    given prior parameters g, a(lpha), b(eta)
    '''   
    posterior = [] 
    for p_a in range(len(possible_p)):
        for p_n in range(len(possible_p)):
            lik_i = U18_likelihood(data,possible_p[p_a],possible_p[p_n])
            prior_i = U18_prior(g,a,b,possible_p[p_a],possible_p[p_n]) 
            posterior.append(lik_i+prior_i)  
    
    return posterior

############################
# Roulette wheel functions #  
############################

def normalize_log_distribution(distribution):
    '''Normalizes a list of log posteriors to make it 
    a (non-log) probability distribution
    '''
    exp_dist = []
    for logp in distribution:
        exp_dist.append(exp(logp))
        #distribution=[exp(d) for d in distribution]
    norm_dist =[]
    for p in exp_dist:
        norm_dist.append(p/sum(exp_dist))
        #distribution=[d/sum(distribution) for d in distribution]
    return norm_dist


def U18_roulette_wheel(g,a,b,data,num_samps):
    '''Generates a random sample of grammars (p_AdjN, p_NumN pairs)
    with probability of selection being proportional to posterior probability
    '''
    post = U18_posterior(g,a,b,data) # calculate posterior given training data and prior parameters
    #post = normalize_log_distribution(post) # normalize --> probability distribution
    post = normalize_log_distribution(post)
    # make a grid of all possible p_AdjN, p_NumN combinations at the granularity specified
    grid_adj = []
    grid_num = []
    for p_a in range(len(possible_p)):
        for p_n in range(len(possible_p)):
            grid_adj.append(possible_p[p_a])
            grid_num.append(possible_p[p_n])
    grid = zip(grid_adj,grid_num) # combine them because posterior probability is for the combination
    
    # samples some grammars!
    grammars = []
    for i in range(0,num_samps):
        r=choice(a=range(0,len(grid)),p=post)   # choose an index from the grid according to it's posterior probability  
        grammars.append(grid[r])
    return grammars

#################################
# Plotting posterior samples... #
#################################

import pylab as plt

def plot_grammars(g_1,g_2,g_3,g_4):
    '''
    Plot grammars sampled from posterior 
    (results of multiple U18_roulette_wheel() calls)
    '''
    col = []
    for i in range(0,4):
        col.append([i for g in range(0,len(g_1))])
    
    plt.title("Sampled grammars")
    plt.xlabel("P(Num-N)");plt.ylabel("P(Adj-N)")
    plt.xlim(0,1);plt.ylim(0,1)
    
    x = [g[1] for g in g_1] + [g[1] for g in g_2] + [g[1] for g in g_3] + [g[1] for g in g_4]  
    y = [g[0] for g in g_1] + [g[0] for g in g_2] + [g[0] for g in g_3] + [g[0] for g in g_4]
    
    plt.scatter(x,y,c=col)
    plt.show()

#######################
# Iterating functions #  
#######################  
def U18_classify(p_AdjN,p_NumN):
    '''Returns pattern type given (p_AdjN,p_NumN) pair.
    '''
    if p_AdjN > 0.5 and p_NumN > 0.5: return 1
    if p_AdjN < 0.5 and p_NumN < 0.5: return 2
    if p_AdjN < 0.5 and p_NumN > 0.5: return 3
    if p_AdjN > 0.5 and p_NumN < 0.5: return 4
    
    else: return 0
    
def U18_produce(p_AdjN,p_NumN):
    '''Returns counts of Adj-N,N-Adj,Num-N,N-Num given (p_AdjN,p_NumN) pair.
    '''
    counts=[]
    AdjN = binomial(n=40,p=p_AdjN) # number of Adj-N out of n trials with p=p_AdjN
    NumN = binomial(n=40,p=p_NumN) # number of Num-N out of n trials with p=p_NumN
    counts.extend([AdjN, 40-AdjN, NumN, 40-NumN])
    return counts

def U18_iterate(starting_data,g,a,b,generations,num_samps):
    '''Iterates from starting data consisting of counts of Adj-N, Num-N 
    Returns number of each pattern type left out of num_samps*4 in each generation. 
    Steps: 
    (1) get samples from each condition given starting_data
    (2) pick random grammar from each set of samples and use to generate new starting_data for that condition
    (3) count the number of each patterns resulting from those samples
    (4) REPEAT
    '''
    
    pattern_tracer=[[],[],[],[]] # value of the function, each sublist tracks count of each pattern type over generations
    
    for gen in range(0,generations):
        patterns_g=[] # accumulator for pattern types in each sample for the current generation
        new_data=[[],[],[],[]] # to be used as starting_data in the subsequent generation
        
        # for each condition, get sample of grammars, generate new training data, count patterns...
        for i in range(0,4):
            print "sampling for condition " + str(i+1) + "..."; sys.stdout.flush()
            samps_i = U18_roulette_wheel(g,a,b,data=starting_data[i],num_samps=num_samps) # get sample of grammars for current condition
            r = random.randint(0,num_samps-1)# pick a random index from samps
            training_g = samps_i[r] # get the grammar at index r 
            print "learner has acquired: " + str(training_g); sys.stdout.flush()
            new_data[i] = U18_produce(training_g[0],training_g[1]) # use grammar to generate new starting data for current condition
            print "passing on:" + str(new_data[i]) + " to next generation..."; sys.stdout.flush()
            # now for each grammar in the sample, classify it and add pattern to the accumulator
            for s in range(0,len(samps_i)):
                patterns_g.append(U18_classify(samps_i[s][0],samps_i[s][1]))
        
        # go through pattern accumulator and count each type for the current generation
        for i in range(0,4): 
            pattern_tracer[i].append(patterns_g.count(i+1)) # add the set of patterns for current condition to list
        print "Frequency of patterns for this generation: " + str(pattern_tracer); sys.stdout.flush()
            
        starting_data=new_data # make the new training data the starting data
    
    return pattern_tracer


####################################
# Plotting counts over generations #
####################################

def plot_counts(counts,generations):
    '''
    Plot count of grammars over generations (output of U18_iterate() call)
    '''
    m = (max(counts[0]),max(counts[1]),max(counts[2]),max(counts[3]))
    lim = max(m)+50
    
    plt.title("Counts of pattern type over generations")
    plt.xlim(1,generations);plt.ylim(0,lim)
    plt.xlabel("Generation");plt.ylabel("Count")
    plt.xticks(range(1,generations+1))
   
    for i in range(0,4):
        plt.plot(range(1,generations+1),counts[i],label='pattern '+str(i+1))
    plt.legend(loc='upper left',fontsize=12)
    plt.show()