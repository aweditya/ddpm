import torch
import pytorch_lightning as pl

class LitDiffusionModel(pl.LightningModule):
    def __init__(self, n_dim=3, n_steps=200, lbeta=1e-5, ubeta=1e-2):
        super().__init__()
        """
        If you include more hyperparams (e.g. `n_layers`), be sure to add that to `argparse` from `train.py`.
        Also, manually make sure that this new hyperparameter is being saved in `hparams.yaml`.
        """
        self.save_hyperparameters()

        """
        Your model implementation starts here. We have separate learnable modules for `time_embed` and `model`.
        You may choose a different architecture altogether. Feel free to explore what works best for you.
        If your architecture is just a sequence of `torch.nn.XXX` layers, using `torch.nn.Sequential` will be easier.
        
        `time_embed` can be learned or a fixed function based on the insights you get from visualizing the data.
        If your `model` is different for different datasets, you can use a hyperparameter to switch between them.
        Make sure that your hyperparameter behaves as expecte and is being saved correctly in `hparams.yaml`.
        """
        self.time_embed = torch.nn.Identity
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_dim + 1, 4*n_dim),
            torch.nn.Linear(4*n_dim, 2*n_dim),
            torch.nn.Linear(2*n_dim, 1)
        )

        """
        Be sure to save at least these 2 parameters in the model instance.
        """
        self.n_steps = n_steps
        self.n_dim = n_dim

        """
        Sets up variables for noise schedule
        """
        [self.alpha_t, self.alpha_bar_t, self.beta_t] = self.init_alpha_beta_schedule(lbeta, ubeta)

    def forward(self, x, t):
        """
        Similar to `forward` function in `nn.Module`. 
        Notice here that `x` and `t` are passed separately. If you are using an architecture that combines
        `x` and `t` in a different way, modify this function appropriately.
        """
        if not isinstance(t, torch.Tensor):
            t = torch.LongTensor([t]).expand(x.size(0))
        t_embed = self.time_embed(t)
        return self.model(torch.cat((x, t_embed), dim=1).float())

    def init_alpha_beta_schedule(self, lbeta, ubeta):
        """
        Set up your noise schedule. You can perhaps have an additional hyperparameter that allows you to
        switch between various schedules for answering q4 in depth. Make sure that this hyperparameter 
        is included correctly while saving and loading your checkpoints.
        """
        beta_t = torch.arange(lbeta, self.n_steps, ubeta)
        alpha_t = 1 - beta_t
        alpha_bar_t = torch.ones(self.n_steps)        
        alpha_bar_t[0] = alpha_t[0]
        for t in range(1, self.n_steps):
            alpha_bar_t[t] = alpha_bar_t[t-1] * alpha_t[t]
        return [alpha_t, alpha_bar_t, beta_t]

    def q_sample(self, x, t):
        """
        Sample from q given x_t.
        """
        return torch.normal(mean=torch.sqrt(1 - self.beta_t[t])*x, std=self.beta_t[t]*torch.eye(n=self.n_dim))

    def p_sample(self, x, t):
        """
        Sample from p given x_t.
        """
        mu_theta = 1/torch.sqrt(self.alpha_t[t]) * (x - self.beta_t[t] / torch.sqrt(1  - self.alpha_bar_t[t]) * self.forward(x, t))
        return torch.normal(mean=mu_theta, std=self.beta_t[t]*torch.eye(n=self.n_dim))

    def training_step(self, batch, batch_idx):
        """
        Implements one training step.
        Given a batch of samples (n_samples, n_dim) from the distribution you must calculate the loss
        for this batch. Simply return this loss from this function so that PyTorch Lightning will 
        automatically do the backprop for you. 
        Refer to the DDPM paper [1] for more details about equations that you need to implement for
        calculating loss. Make sure that all the operations preserve gradients for proper backprop.
        Refer to PyTorch Lightning documentation [2,3] for more details about how the automatic backprop 
        will update the parameters based on the loss you return from this function.

        References:
        [1]: https://arxiv.org/abs/2006.11239
        [2]: https://pytorch-lightning.readthedocs.io/en/stable/
        [3]: https://www.pytorchlightning.ai/tutorials
        """
        n_samples, _ = batch.size(0)
        t = torch.randint(low=0, high=self.n_steps-1, size=[n_samples])
        epsilon = torch.normal(size=n_samples)

        batch_updated = torch.sqrt(self.alpha_bar_t[t]) * batch + torch.sqrt(1 - self.alpha_bar_t[t]) * epsilon
        epsilon_theta = self.forward(batch_updated, t)
        loss =  torch.nn.MSELoss(reduction='sum')
        return loss(epsilon_theta, epsilon)

    def sample(self, n_samples, progress=False, return_intermediate=False):
        """
        Implements inference step for the DDPM.
        `progress` is an optional flag to implement -- it should just show the current step in diffusion
        reverse process.
        If `return_intermediate` is `False`,
            the function returns a `n_samples` sampled from the learned DDPM
            i.e. a Tensor of size (n_samples, n_dim).
            Return: (n_samples, n_dim)(final result from diffusion)
        Else
            the function returns all the intermediate steps in the diffusion process as well 
            i.e. a Tensor of size (n_samples, n_dim) and a list of `self.n_steps` Tensors of size (n_samples, n_dim) each.
            Return: (n_samples, n_dim)(final result), [(n_samples, n_dim)(intermediate) x n_steps]
        """
        pass

    def configure_optimizers(self):
        """
        Sets up the optimizer to be used for backprop.
        Must return a `torch.optim.XXX` instance.
        You may choose to add certain hyperparameters of the optimizers to the `train.py` as well.
        In our experiments, we chose one good value of optimizer hyperparameters for all experiments.
        """
        return torch.optim.Adam
