import numpy as np

def perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)

    # Your implementation here
    th=np.zeros(shape=(data.shape[0],1))
    th_0=np.zeros(shape=(1,1))
    for t in range(0,T):
        for i in range(0,data.shape[1]):
            if (labels[0][i]*(th_0 + np.dot(th.T,data[:,i:i+1])))<=0:
                th +=  labels[0][i]*data[:,i:i+1]
                th_0 += labels[0][i]
    
    return (th,th_0)


def averaged_perceptron(data, labels, params={}, hook=True):
    # if T not in params, default to 100
    T = params.get('T', 100)

    # Your implementation here
    th=np.zeros(shape=(data.shape[0],1))
    ths=np.zeros(shape=(data.shape[0],1))
    th_0=np.zeros(shape=(1,1))
    th_0s=np.zeros(shape=(1,1))
    counter=0
    for t in range(T):
        counter+=1
        for i in range(data.shape[1]):
            if (labels[:,i]*(th_0 + np.dot(th.T,data[:,i:i+1])))<=0:
                th +=  labels[0][i]*data[:,i:i+1]
                th_0 += labels[0][i]
            ths += th
            th_0s += th_0

    return (ths/(data.shape[1]*counter),th_0s/(data.shape[1]*counter))


def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th= learner(data_train,labels_train)[0]
    th0= learner(data_train,labels_train)[1]
    return (score(data_test,labels_test,th,th0)/data_test.shape[1])

def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    acc=[]
    for i in range(it):
        train=data_gen(n_train)
        test=data_gen(n_test)
        acc.append(eval_classifier(learner, train[0], train[1], test[0], test[1]))
    return (np.mean(acc))

def xval_learning_alg(learner, data, labels, k):
    #cross validation of learning algorithm
    scores=[]
    for j in range(k):
        split=np.array_split(data,k,axis=1)
        labs= np.array_split(labels,k,axis=1)
        test_d= split[j]
        test_l= labs[j]
        train_d= np.concatenate(split[:j]+split[j+1:],axis=1)
        train_l= np.concatenate(labs[:j]+labs[j+1:],axis=1)
        scores.append(eval_classifier(learner, train_d, train_l,
        test_d, test_l))
    return np.mean(scores)