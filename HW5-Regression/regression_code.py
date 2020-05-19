import numpy as np

# Enter an expression to compute and set th to the optimal theta
th = np.linalg.inv((X@np.transpose(X)))@X@np.transpose(Y)

def d_lin_reg_th(x, th, th0):
    """
    Parameters:
        x is d by n : input data
        th is d by 1 : weights
        th0 is 1 by 1 or scalar
    Returns:
        d by n array : gradient of lin_reg(x, th, th0) with respect to th
    """
    return x
    
def d_square_loss_th(x, y, th, th0):
    """
    Parameters:
        x is d by n : input data
        y is 1 by n : output regression values
        th is d by 1 : weights
        th0 is 1 by 1 or scalar
    Returns:
        d by n array : gradient of square_loss(x, y, th, th0) with respect to th.
    
    This function should be a one-line expression that uses lin_reg and
    d_lin_reg_th.
    """
    return 2*(lin_reg(x, th, th0)-y)*(d_lin_reg_th(x, th, th0))

def d_mean_square_loss_th(x, y, th, th0):
    """
    Parameters:
        Same as above
    Returns:
        d by 1 array : gradient of mean_square_loss(x, y, th, th0) with respect to th.
    
    This function should be a one-line expression that uses d_square_loss_th.
    """
    return np.mean(d_square_loss_th(x, y, th, th0), axis = 1).reshape(-1,1)


def d_lin_reg_th0(x, th, th0):
    """
    Parameters:
        x is d by n : input data
        th is d by 1 : weights
        th0 is 1 by 1 or scalar
    Returns:
        1 by n array : gradient of lin_reg(x, th, th0) with respect to th0
    """
    return np.ones(x.shape[1]).reshape(1,x.shape[1])
    
def d_square_loss_th0(x, y, th, th0):
    """
    Parameters:
        x is d by n : input data
        y is 1 by n : output regression values
        th is d by 1 : weights
        th0 is 1 by 1 or scalar
    Returns:
        1 by n array : gradient of square_loss(x, y, th, th0) with respect to th0.
    
    This function should be a one-line expression that uses lin_reg and
    d_lin_reg_th0.
    """
    return -2*(y-lin_reg(x, th, th0))*d_lin_reg_th0(x, th, th0)

def d_mean_square_loss_th0(x, y, th, th0):
    """
    Parameters:
        Same as above
    Returns:
        1 by 1 array : gradient of mean_square_loss(x, y, th, th0) with respect to th0.
    
    This function should be a one-line expression that uses d_square_loss_th0.
    """
    return np.mean(d_square_loss_th0(x, y, th, th0), axis = 1, keepdims = True)


def d_ridge_obj_th(x, y, th, th0, lam):
    return np.mean(d_square_loss_th(x, y, th, th0), axis = 1).reshape(-1,1) + 2*lam*th

def d_ridge_obj_th0(x, y, th, th0, lam):
    return np.mean(d_square_loss_th0(x, y, th, th0), axis = 1, keepdims = True) 


# for sgd:
# X: a standard data array (d by n)
# y: a standard labels row vector (1 by n)
# J: a cost function whose input is a data point (a column vector), a label (1 by 1) and a weight vector w (a column vector) (in that order), and which returns a scalar.
# dJ: a cost function gradient (corresponding to J) whose input is a data point (a column vector), a label (1 by 1) and a weight vector w (a column vector) (also in that order), and which returns a column vector.
# w0: an initial value of weight vector ww, which is a column vector.
# step_size_fn: a function that is given the (zero-indexed) iteration index (an integer) and returns a step size.
# max_iter: the number of iterations to perform

# w: the value of the weight vector at the final step
# fs: the list of values of JJ found during all the iterations
# ws: the list of values of intermediate ww found during all the iterations

def sgd(X, y, J, dJ, w0, step_size_fn, max_iter):
    fs=[]
    ws=[]
    w=w0
    for i in range(max_iter):
        ind=np.random.randint(X.shape[1])
        fs.append(J(X[:,ind].reshape(-1,1),y[0][ind],w))
        ws.append(w)
        w=w-step_size_fn(i)*dJ(X[:,ind].reshape(-1,1),y[0][ind],w)
    return (w,fs,ws)
    
