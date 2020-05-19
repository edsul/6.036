import numpy as np

def dReLU_dz(z):
    n=z.shape[0]
    a=np.zeros(shape=(n,1))
    for i in range(n):
        a[i]= max(0,z[i])
    for i in range(n):
        if a[i]>0:
            a[i]=1
    return a
        
    
#HW7#
class Module:
    def sgd_step(self, lrate): pass  # For modules w/o weights

class Linear(Module):
    def __init__(self, m, n):
        self.m, self.n = (m, n)  # (in size, out size)
        self.W0 = np.zeros([self.n, 1])  # (n x 1)
        self.W = np.random.normal(0, 1.0 * m ** (-.5), [m, n])  # (m x n)

    def forward(self, A):
        self.A = A   # (m x b)  Hint: make sure you understand what b stands for
        return self.W.T@A +self.W0  # Your code (n x b)

    def backward(self, dLdZ):  # dLdZ is (n x b), uses stored self.A
        self.dLdW = self.A@dLdZ.T      # Your code
        self.dLdW0 = dLdZ     # Your code
        return self.W@dLdZ      # Your code: return dLdA (m x b)

    def sgd_step(self, lrate):  # Gradient descent step
        self.W = self.W - lrate*self.dLdW           # Your code
        self.W0 = self.W0 - np.sum(lrate*self.dLdW0,axis=1).reshape(-1,1) 

class Tanh(Module):            # Layer activation
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):    # Uses stored self.A
        return dLdA*(1-self.A**2)     # Your code: return dLdZ with dimensions (?, b)

class ReLU(Module):              # Layer activation
    def forward(self, Z):
        self.A = np.maximum(0,Z)          # Your code: (?, b)
        return self.A

    def backward(self, dLdA):    # uses stored self.A
        return dLdA*np.sign(self.A)
             # Your code: return dLdZ (?, b) 

class SoftMax(Module):           # Output activation
    def forward(self, Z):
        return np.exp(Z)/np.sum(np.exp(Z),axis=0)             # Your code: (?, b)

    def backward(self, dLdZ):    # Assume that dLdZ is passed in
        return dLdZ

    def class_fun(self, Ypred):  # Return class indices
        return np.argmax(Ypred,axis=0)      # Your code: A 1D vector (b, ) 
        
class NLL(Module):       # Loss
    def forward(self, Ypred, Y):
        self.Ypred = Ypred
        self.Y = Y
        return -np.sum((Y*np.log(Ypred))) # Your code

    def backward(self):  # Use stored self.Ypred, self.Y
        return self.Ypred-self.Y      # Your code


class Sequential:
    def __init__(self, modules, loss):            # List of modules, loss module
        self.modules = modules
        self.loss = loss

    def sgd(self, X, Y, iters=100, lrate=0.005):  # Train
        D, N = X.shape
        sum_loss = 0
        for it in range(iters):
            r=np.random.randint(N)
            Xt=X[:,r].reshape(-1,1)
            Yt=Y[:,r].reshape(-1,1)  
            Ypred= self.forward(Xt)
            loss= self.loss.forward(Ypred,Yt)
            dLdZ= self.loss.backward()
            self.backward(dLdZ)
            self.sgd_step(lrate)
            self.print_accuracy(it,X,Y,loss,every=250)

    def forward(self, Xt):                        # Compute Ypred
        for m in self.modules: Xt = m.forward(Xt)
        return Xt

    def backward(self, delta):                    # Update dLdW and dLdW0
        # Note reversed list of modules
        for m in self.modules[::-1]: delta = m.backward(delta)

    def sgd_step(self, lrate):                    # Gradient descent step
        for m in self.modules: m.sgd_step(lrate)

    def print_accuracy(self, it, X, Y, cur_loss, every=250):
        # Utility method to print accuracy on full dataset, should
        # improve over time when doing SGD. Also prints current loss,
        # which should decrease over time. Call this on each iteration
        # of SGD!
        if it % every == 1:
            cf = self.modules[-1].class_fun
            acc = np.mean(cf(self.forward(X)) == cf(Y))
            print('Iteration =', it, '\tAcc =', acc, '\tLoss =', cur_loss, flush=True)
