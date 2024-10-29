
class NoamOpt:
    """
    Optimizer wrapper that implements a learning rate schedule.

    This schedule varies the learning rate during training as follows:

    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

    This results in a learning rate that increases linearly during a warmup period
    and then decreases proportionally to the inverse square root of the step number.
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        '''
        Input:   
        model_size : int
            The size of the model (e.g., hidden dimension size) which influences the learning rate scale.
        
        factor : float
            A scaling factor for the learning rate, providing additional control over the learning rate magnitude.
        
        warmup : int
            The number of warmup steps during which the learning rate increases linearly.
        
        optimizer : torch.optim.Optimizer
            The underlying optimizer whose learning rate is being controlled by this schedule.
        '''
        self.optimizer = optimizer
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup

        self._step = 0
        self._rate = 0

    def step(self):
        """
        Update parameters and rate and step the optimizer
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):

        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))