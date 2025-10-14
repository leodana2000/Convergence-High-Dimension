import numpy as np
import torch as t

from model_data_train import One_hidden_layer, generate_data_exp1, train

def nb_neurons(n:int, eps:float) -> int:
    """ Corresponds to the number of neurons necessary to have a good initialization with probability at least {eps}. """
    return int(np.floor(np.log(n/eps)/np.log(4/3)))+1

def nb_epochs(n:int, p:int, lr:float) -> int:
    """ Corresponds to 1.5 the phase transition threshold. """
    return int(1.5*np.sqrt(n*p)*np.log(n*p)/(4*lr)+1)


def reg_lin(X, Y):
    """ Fits a linear regression. """
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)

    a = np.sum((X-mean_X)*(Y-mean_Y))/np.sum((X-mean_X)**2)
    b = mean_Y - a*mean_X

    eps = Y - a*mean_X - b
    max_eps = np.max(np.abs(eps))
    return (a,b,max_eps)


def sigmoid(x):
    return 1/(1+np.exp(-x))

def fit_sigmoid(X: list[float], Y: list[float]) -> float:
    """ Fits a sigmoid function to find the transition threshold. """
    x = t.tensor(X).unsqueeze(0)
    x_min = t.min(x).item()
    x_max = t.max(x).item()
    k = t.linspace(x_min, x_max, 10000).unsqueeze(1)
    y = t.tensor(Y).unsqueeze(0)
    loss = t.sum((y-sigmoid(-(x - k)))**2, dim=1)
    idx = t.argmin(loss).item()
    K = k[idx, 0].item()

    return K

def do_list(min, max, step) -> list[int]:
    return [x for x in range(min, max + step, step)]

def do_train(
    d: int, n: int, p: int,
    lr: float, repetition: int
) -> tuple[float, float]:

    epoch = nb_epochs(n, p, lr) # This value corresponds to 1.5 times the phase transition threshold.
    loss = 0.
    proba = 0.
    for _ in range(repetition):
        model = One_hidden_layer(emb_dim=d, hid_dim=p)
        data = generate_data_exp1(d, n)
        Loss = train(model, data, lr, epoch, verbose=False)

        if Loss[-1] <= p/(2*n):
            loss += 0.
            proba += 1/repetition
        else:
            loss += Loss[-1]/(repetition*Loss[0])
            proba += 0.

    return loss, proba
