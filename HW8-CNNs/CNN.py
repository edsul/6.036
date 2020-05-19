import numpy as np 

# X: a standard data array (d by n)
# y: a standard labels row vector (1 by n)
# iters: the number of updates to perform on weights WW
# lrate: the learning rate used
# K: the mini-batch size to be used

import math 

class Sequential:
    def __init__(self, modules, loss):            
        self.modules = modules
        self.loss = loss

    def mini_gd(self, X, Y, iters, lrate, notif_each=None, K=10):
        D, N = X.shape

        np.random.seed(0)
        num_updates = 0
        indices = np.arange(N)
        while num_updates < iters:

            np.random.shuffle(indices)
            X = X[:,indices]  # Your code
            Y = Y[:,indices]  # Your code
            
            for j in range(math.floor(N/K)):
                if num_updates >= iters: break

                # Implement the main part of mini_gd here
                Xt = X[:,(j*K):(j+1)*K] # Your code
                Yt = Y[:,(j*K):(j+1)*K] # Your code

                # The rest of this function should be similar to your
                # implementation of Sequential.sgd in HW 7
                # Your code
                Ypred= self.forward(Xt)
                loss= self.loss.forward(Ypred,Yt)
                dLdZ= self.loss.backward()
                self.backward(dLdZ)
                self.sgd_step(lrate)
                num_updates += 1

    def forward(self, Xt):                        
        for m in self.modules: Xt = m.forward(Xt)
        return Xt

    def backward(self, delta):                   
        for m in self.modules[::-1]: delta = m.backward(delta)

    def sgd_step(self, lrate):    
        for m in self.modules: m.sgd_step(lrate)


class BatchNorm(Module):    
    def __init__(self, m):
        np.random.seed(0)
        self.eps = 1e-20
        self.m = m  # number of input channels
        
        # Init learned shifts and scaling factors
        self.B = np.zeros([self.m, 1]) # m x 1
        self.G = np.random.normal(0, 1.0 * self.m ** (-.5), [self.m, 1]) # m x 1
        
    # Works on m x b matrices of m input channels and b different inputs
    def forward(self, A):# A is m x K: m input channels and mini-batch size K
        # Store last inputs and K for next backward() call
        self.A = A
        self.K = A.shape[1]
        
        self.mus = np.mean(A,axis=1).reshape(-1,1)  # Your Code
        self.vars = np.var(A,axis=1).reshape(-1,1)  # Your Code

        # Normalize inputs using their mean and standard deviation
        self.norm = (A-self.mus)/(np.sqrt(self.vars)+self.eps)  # Your Code
            
        # Return scaled and shifted versions of self.norm
        return self.G*self.norm + self.B  # Your Code

    def backward(self, dLdZ):
        # Re-usable constants
        std_inv = 1/np.sqrt(self.vars+self.eps)
        A_min_mu = self.A-self.mus
        
        dLdnorm = dLdZ * self.G
        dLdVar = np.sum(dLdnorm * A_min_mu * -0.5 * std_inv**3, axis=1, keepdims=True)
        dLdMu = np.sum(dLdnorm*(-std_inv), axis=1, keepdims=True) + dLdVar * (-2/self.K) * np.sum(A_min_mu, axis=1, keepdims=True)
        dLdX = (dLdnorm * std_inv) + (dLdVar * (2/self.K) * A_min_mu) + (dLdMu/self.K)
        
        self.dLdB = np.sum(dLdZ, axis=1, keepdims=True)
        self.dLdG = np.sum(dLdZ * self.norm, axis=1, keepdims=True)
        return dLdX

    def sgd_step(self, lrate):
        self.B = self.B-lrate*self.dLdB  # Your Code
        self.G = self.G-lrate*self.dLdG  # Your Code
        return
