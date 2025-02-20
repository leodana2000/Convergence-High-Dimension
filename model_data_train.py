import torch as t
import numpy as np
from typing import Tuple
from tqdm import tqdm


class One_hidden_layer(t.nn.Module):
    """Instantiate a two-layer NN with ReLU."""
    def __init__(self, emb_dim: int, hid_dim: int, biases: bool = False) -> None:
        super().__init__()

        self.computations = t.nn.Sequential(
            t.nn.Linear(emb_dim, hid_dim, bias=biases, dtype=t.float),
            t.nn.ReLU(inplace=True),
            t.nn.Linear(hid_dim, 1, bias=biases, dtype=t.float),
        )

        norm_init = t.norm(self.computations[0].weight.data, dim=1, keepdim=True).transpose(0,1)
        a_j = t.tensor(2*np.random.binomial(1, 0.5, size=norm_init.shape)-1)*(norm_init+t.tensor(np.random.exponential(1, size=norm_init.shape)))
        self.computations[2].weight = t.nn.Parameter(a_j.to(t.float))

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.computations(x)/self.hid_dim
    
    def is_well_init(self, data: Tuple[t.Tensor, t.Tensor]) -> bool:
        flag = True
        WX = self.computations[0](data[0])
        for i, y in enumerate(data[1]):
            drapeau = False
            for j, a_j in enumerate(self.computations[2].weight[0]):
                drapeau = drapeau or (((a_j*y).item() >0) and (WX[i,j].item() > 0))
            flag = flag and drapeau
        return flag


def generate_data_exp1(emb_dim: int, num_data: int) -> Tuple[t.Tensor, t.Tensor]:
    """
    Returns num_data vectors of dimension emb_dim and their real label.\\
    X/||X|| ∼ U(S^d-1)\\
    ||X|| ∼ U([1,2])\\
    Y/|Y| ∼ U({-1,1})\\
    |Y| ∼ U([1,2])
    """
    Gaussians = t.tensor(np.random.normal(0, 1/emb_dim, size=(num_data, emb_dim)))
    Isotrope = Gaussians/t.norm(Gaussians, dim=1, keepdim=True)
    X = (Isotrope*t.tensor(np.random.uniform(1, 2, size=(num_data, 1)))).to(t.float)
    abs_Y = t.tensor(np.random.uniform(1, 2, size=(num_data, 1)))
    Rademacher = 2*t.tensor(np.random.binomial(1, 0.5, size=(num_data, 1))) - 1
    Y = (abs_Y*Rademacher).squeeze()
    return (X,Y)

def generate_data_exp2(emb_dim: int, num_data: int) -> Tuple[t.Tensor, t.Tensor]:
    """
    Returns num_data vectors of dimension emb_dim and their real label.\\
    X/||X|| ∼ O(S^d-1) # Orthogonals on the sphere\\
    ||X|| ∼ U([1,2])\\
    Y/|Y| ∼ U({-1,1})\\
    |Y| ∼ U([1,2])
    """
    Orthogonals = t.eye(emb_dim)[:num_data,:]
    X = Orthogonals*t.tensor(np.random.uniform(1, 2, size=(num_data, 1)))
    abs_Y = t.tensor(np.random.uniform(1, 2, size=(num_data, 1)))
    Rademacher = 2*t.tensor(np.random.binomial(1, 0.5, size=(num_data, 1))) - 1
    Y = (abs_Y*Rademacher).squeeze()
    return (X.to(t.float),Y.to(t.float))


def train(model: One_hidden_layer, data: Tuple[t.Tensor, t.Tensor], lr: float, epochs: int, cv_threshold: float=1., verbose:bool=True) -> list:
    optimizer = t.optim.SGD(model.parameters(), lr, 
                            momentum=0, 
                            dampening=0, 
                            weight_decay=0, 
                            nesterov=False) # Vanilla GD
    num_data = len(data[0])
    loss_norm = num_data
    num_neuron = model.hid_dim

    Loss = []
    for _ in tqdm(range(epochs), disable=not verbose):
        residuals = (data[1] - model(data[0]).squeeze())
        loss = num_neuron*t.mean((residuals)**2)/2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        Loss.append(loss.item())
        if loss < num_neuron/(2*loss_norm)*cv_threshold:
            break # Early stopping: if the loss is small enough, it will reach 0.

    return Loss