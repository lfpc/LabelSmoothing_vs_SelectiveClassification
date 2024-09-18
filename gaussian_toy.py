import torch

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def generate_random_orthogonal_vectors(N,dimensions):
    random_matrix = torch.randn(dimensions,dimensions)
    Q = torch.linalg.qr(random_matrix)[0][:N]
    return torch.nn.functional.normalize(Q, p=2, dim=1)

def generate_random_probability_vector(size):
    # Generate a random vector from the Dirichlet distribution
    alpha = torch.ones(size)  # Parameter alpha for the Dirichlet distribution
    probability_vector = torch.distributions.Dirichlet(alpha).sample()
    return probability_vector


class MixGaussian():
    def __init__(self,center:torch.tensor,
                 py:torch.tensor,
                 std_diag:torch.tensor) -> None:
        self.center = torch.as_tensor(center,device = dev,dtype = float)
        self.py = torch.as_tensor(py,device = dev)
        assert self.py.sum().isclose(torch.tensor(1.))
        self.classes_dist = torch.distributions.Categorical(self.py)
        self.std_diag = torch.as_tensor(std_diag,device = dev).view(-1,1)#torch.diag(std_diag)
    def __sample_classes(self,N:int):
        return self.classes_dist.sample((N,))
    def sample(self,N:int):
        classes = self.__sample_classes(N)
        return torch.normal(mean=self.center[classes], std=self.std_diag[classes]).cpu(),classes.flatten().to(int).cpu()
    def log_likelihood(self,x:torch.tensor) -> torch.tensor:
        std = self.std_diag.view(-1)
        coeff = (std * torch.sqrt(torch.tensor(2 * torch.pi))).log()
        Z = (x.unsqueeze(1) - self.center).div(std)
        return (Z.pow(2).mul(-0.5)-coeff).sum(-1)
    def log_posterior(self,x:torch.tensor) -> torch.tensor:
        joint_likelihood = self.log_likelihood(x)+self.py.log()
        return joint_likelihood - joint_likelihood.logsumexp(-1).unsqueeze(-1)
    
if __name__ == '__main__':
    dimensions = 128
    n_classes = 100
    centers =  generate_random_orthogonal_vectors(n_classes,dimensions)#torch.rand(n_classes,dimensions)
    Py = generate_random_probability_vector(n_classes)#torch.ones(n_classes)/n_classes#
    std = 0.3*torch.ones(dimensions) #torch.rand(dimensions)

    dist = MixGaussian(centers,
                   Py,
                   std)
    
    N  = 10000

    data_train = None#dist.sample(N)
    data_val = dist.sample(N)
    data_test = dist.sample(N)

    metric = torch.nn.KLDivLoss(log_target = True, reduction = 'batchmean')
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.0)

    class Model(torch.nn.Module):
        def __init__(self, dim_input,dim_output, **kwargs) -> None:
            super().__init__(**kwargs)
            self.W = torch.nn.Linear(dim_input,dim_output)
        def forward(self,x:torch.tensor):
            return self.W(x)

    model_nll = Model(dimensions,n_classes).to(dev)