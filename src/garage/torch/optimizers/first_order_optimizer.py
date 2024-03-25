from torch.optim import Adam

class FirstOrderOptimizer(Adam):
    """First order optimier.

    Performs (stochastic) gradient descent, possibly using fancier methods like
    ADAM etc.

    Args:
        optimizer (torch.optim): Optimizer to be used
        params (iterable): Iterable of parameters to optimize.
        learning_rate (float): learning rate.
            learning rates are our main interest parameters to tune optimizers.
        max_optimization_epochs (int): Maximum number of epochs for update.

    """
    def __init__(
        self,
        network_params,
        learning_rate=5e-4,
        batch_size=32,
        max_optimization_epochs=1000,
    ):
        super().__init__(network_params(), learning_rate)
        self._learning_rate = learning_rate
        self._max_optimization_epochs = max_optimization_epochs
        self._batch_size = batch_size

