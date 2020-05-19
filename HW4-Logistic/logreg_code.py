import numpy as np
def gd(f, df, x0, step_size_fn, num_steps):
    for i in range(num_steps):
        x0 = x0 - step_size_fn(i)*(df(x0))
        
    return x0,f(x0)

def num_grad(f, delta=0.001):
    def df(x):
        m=x.shape[0]
        res= np.zeros(shape=(m,1))
        for i in range(m):
            delt=np.zeros(shape=(m,1))
            delt[i] = delta
            res[i]= (f(x+delt)-f(x-delt))/(2*delta)
        return res
    return df

def minimize(f, x0, step_size_fn, num_steps):
    """
    Parameters:
      See definitions in part 1
    Returns:
      same output as gdf=d
    """
    th=x0
    df=num_grad(f,delta=0.001)
    return gd(f, df, th, step_size_fn, num_steps)

# x is a column vector
# returns a vector of the same shape as x
def sigmoid(x):
    return 1/(1+np.exp(-x))

# X is dxn, y is 1xn, th is dx1, th0 is 1x1
# returns (1,n) the nll loss for each data point given th and th0 
def nll_loss(X, y, th, th0):
    a= th.T@X + th0
    sig_a= sigmoid(a)
    l= -(y*np.log(sig_a) + (1-y)*np.log(1-sig_a))
    return l

# X is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
# returns (float) the llc objective over the dataset
def llc_obj(X, y, th, th0, lam):
    a= th.T@X + th0
    sig_a= sigmoid(a)
    l= -(y*np.log(sig_a) + (1-y)*np.log(1-sig_a))
    return np.sum(l)/(l.shape[1]) + lam*np.linalg.norm(th)**2


########


# returns (1,1) the gradient of sigmoid with respect to x
def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

# returns (d,n) the gradient of nll_loss(X, y, th, th0) with respect to th for each data point
def d_nll_loss_th(X, y, th, th0):
    a= th.T@X + th0
    sig_a= sigmoid(a)
    return (sig_a-y)*X

# returns (1,n) the gradient of nll_loss(X, y, th, th0) with respect to th0
def d_nll_loss_th0(X, y, th, th0):
    a= th.T@X + th0
    sig_a= sigmoid(a)
    return sig_a-y

# returns (d,1) the gradient of llc_obj(X, y, th, th0) with respect to th
def d_llc_obj_th(X, y, th, th0, lam):
    b=d_nll_loss_th(X, y, th, th0)
    b_s=np.sum(b,axis=1).reshape(th.shape[0],-1)
    return (1/X.shape[1])*b_s + 2*lam*th
    
# returns (1,1) the gradient of llc_obj(X, y, th, th0) with respect to th0
def d_llc_obj_th0(X, y, th, th0, lam):
    a=d_nll_loss_th0(X, y, th, th0)
    a_s= (1/X.shape[1])*np.sum(a).reshape(1,1)
    return a_s

# returns (d+1, 1) the full gradient as a single vector (which includes both th, th0)
def llc_obj_grad(X, y, th, th0, lam):
    a=d_llc_obj_th(X, y, th, th0, lam)
    b=d_llc_obj_th0(X, y, th, th0, lam)
    return np.vstack((a,b))


def llc_min(data, labels, lam):
    """
    Parameters:
        data: dxn
        labels: 1xn
        lam: scalar
    Returns:
        same output as gd
    """
    
    th = np.zeros(shape=(data.shape[0],1))
    th0 = np.zeros(shape=(1,1))
    ths= np.vstack((th,th0))
    f = lambda x: llc_obj(data, labels, x[:-1], x[-1], lam)
    df = lambda x: llc_obj_grad(data, labels, x[:-1], x[-1], lam)
    
    def llc_min_step_size_fn(i):
       return 2/(i+1)**0.5

    return gd(f,df,ths,llc_min_step_size_fn,num_steps=10)
