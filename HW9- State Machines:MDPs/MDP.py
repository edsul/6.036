import numpy as np
class SM:
    start_state = None

    def transduce(self, input_seq):
        '''input_seq: a list of inputs to feed into SM
           returns:   a list of outputs of SM'''
        res=[]
        for x in input_seq:
            new_state= self.transition_fn(self.start_state,x)
            self.start_state = new_state
            res.append(self.output_fn(new_state))
        return res

class Binary_Addition(SM):
    start_state = (0,0) # Change

    def transition_fn(self, s, x):
        # Your code here
        if x[0] + x[1] == 2:
            if s[0]>0:
                return (1,1)
            return (1,0)
        elif x[0] + x[1] == 0:
            if s[0]>0:
                return (0,1)
            return (0,0)
        else:
            if s[0]>0:
                return (1,0)
            return (0,1)
        
    def output_fn(self, s):
        # Your code here
        return s[-1]
    
class Binary_Addition(SM):
    start_state = (0,0) # Change

    def transition_fn(self, s, x):
        # Your code here
        if x[0] + x[1] == 2:
            if s[0]>0:
                return (1,1)
            return (1,0)
        elif x[0] + x[1] == 0:
            if s[0]>0:
                return (0,1)
            return (0,0)
        else:
            if s[0]>0:
                return (1,0)
            return (0,1)
        
    def output_fn(self, s):
        # Your code here
        return s[-1]
    
class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2, start_state):
        self.Wsx=Wsx
        self.Wss=Wss
        self.Wo=Wo
        self.Wss_0=Wss_0
        self.Wo_0=Wo_0
        self.f1=f1
        self.f2=f2
        self.start_state=start_state
        pass
    def transition_fn(self, s, x):
        # Your code here
        return self.f1(self.Wss@s + self.Wsx@x + self.Wss_0)
        
    def output_fn(self, s):
        # Your code here
        return self.f2(self.Wo@s + self.Wo_0)

#Accumulator RNN
# Wsx = np.array([[1]])          # Your code here
# Wss = np.array([[1]])              # Your code here
# Wo = np.array([[1]])           # Your code here
# Wss_0 = np.array([[0]])             # Your code here
# Wo_0 = np.array([[0]])         # Your code here
# f1 = lambda x : x# Your code here, e.g. lambda x : x
# f2 = lambda x: np.sign(x)              # Your code here
# start_state =  np.array([[0]])     # Your code here
# acc_sign = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2, start_state)

#Autoregression RNN
# Wsx =  np.zeros(shape=(3,1)) # Your code here
# Wss = np.array([[1,-2,3],[1,0,0],[0,1,0]]) # Your code here
# Wo =  np.array([[1,0,0]])     # Your code here
# Wss_0 =  np.zeros(shape=(3,1))         # Your code here
# Wo_0 = 0       # Your code here
# f1 = lambda x: x            # Your code here, e.g. lambda x : x
# f2 = lambda x: x              # Your code here
# start_state = np.array([[-2,0,0]]).T #(1,3)    # Your code here
# auto = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2, start_state)

def value(q, s):
    """ Return Q*(s,a) based on current Q

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q_star = value(q,0)
    >>> q_star
    10
    """
    res=[]
    for a in q.actions:
        res.append(q.get(s,a))
    return max(res)

def greedy(q, s):
    """ Return pi*(s) based on a greedy strategy.

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q.set(1, 'b', 2)
    >>> greedy(q, 0)
    'c'
    >>> greedy(q, 1)
    'b'
    """
    # Your code here
    res=[]
    for a in (q.actions):
        res.append(q.get(s,a))
    index=np.argmax(res)
    return q.actions[index]
        
def epsilon_greedy(q, s, eps = 0.5):
    """ Returns an action.
    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q.set(1, 'b', 2)
    >>> eps = 0.
    >>> epsilon_greedy(q, 0, eps) #greedy
    'c'
    >>> epsilon_greedy(q, 1, eps) #greedy
    'b'
    """
    if random.random() < eps:# True with prob eps, random action
        # Your code here
        return uniform_dist(q.actions).draw()
    else:
        return greedy(q,s) 
        # Your code here


##
# Make sure to copy the Q function between iterations, e.g. new_q = q.copy(), so that you are only using Q-values from the previous iteration.
# The q parameter contains the initialization of the Q function.
# The value function is already defined.
# Use mdp class definitions to get
# the reward function: reward_fn,
# the discount factor: discount_factor,
# transition model: transition_model,
# expectation of the Q-values over a distribution: transition_model(s,a).expectation.

def value_iteration(mdp, q, eps = 0.01, max_iters=1000):
    # Your code here
    count=0
    states=q.states
    actions=q.actions
    while count<=max_iters:
        count+=1 
        q_old=q.copy()
        for s in states:
            for a in actions:
                dist=mdp.transition_model(s,a)
                probs= dist.getAllProbs()
                sum=0
                for s_,p in probs:
                    x = p*value(q_old,s_)
                    sum+=x
                q.set(s,a,mdp.reward_fn(s,a)+mdp.discount_factor*sum)
        _max=-9**99
        for s in states:
            for a in actions:
                _max=np.maximum(_max,np.abs(q_old.get(s,a)- q.get(s,a)))
        if _max<eps:
            break
        
    return q         


def q_em(mdp, s, a, h):
    # Your code here
    if h==0:
        return 0
    else:
        sum=0
        dist=mdp.transition_model(s,a)
        probs= dist.getAllProbs()
        for st,p in probs:
            max_= -10**9
            for a in mdp.actions:
                x=q_em(mdp, st, a, h-1)
                max_= np.maximum(max_,x)
            sum+=p*max_
        return mdp.reward_fn(s,a)+mdp.discount_factor*sum
                