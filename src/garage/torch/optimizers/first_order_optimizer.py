import torch
from torch.utils.data import TensorDataset, DataLoader


class FirstOrderOptimizer:
    """First order optimizer for PyTorch models, closely following the original structure.

    Args:
        model (torch.nn.Module): The model to optimize.
        optimizer_class (class): The class of optimizer to use from torch.optim.
        learning_rate (float): The learning rate for the optimizer.
        max_optimization_epochs (int): The maximum number of epochs for optimization.
        tolerance (float): The tolerance for early stopping based on loss improvement.
        batch_size (int): The batch size for optimization.
        callback (callable): Optional function to call after each epoch.
        verbose (bool): If true, prints log messages during optimization.
        name (str): The name for this optimizer instance, used in logging.
    """

    def __init__(self, model, optimizer_class=torch.optim.Adam,
                 learning_rate=1e-3,
                 max_optimization_epochs=1000, tolerance=1e-6, batch_size=32,
                 callback=None, verbose=False, name='FirstOrderOptimizer'):
        self._model = model
        self._optimizer_class = optimizer_class
        self._learning_rate = learning_rate
        self._max_optimization_epochs = max_optimization_epochs
        self._tolerance = tolerance
        self._batch_size = batch_size
        self._callback = callback
        self._verbose = verbose
        self._name = name
        # Initializing placeholders for internal use
        self.optimizer = self._optimizer_class(model.parameters(),
                                                lr=self._learning_rate)
        self._loss_function = None

    def update_opt(self, loss_function):
        """Sets up the optimizer and loss function based on provided arguments.

        Args:
            loss_function (callable): The loss function to be used for optimization.
        """
        self._loss_function = loss_function


    def optimize(self, inputs, targets):
        """Executes the optimization process.

        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target data.
        """
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=self._batch_size,
                                shuffle=True)

        prev_loss = float('inf')
        for epoch in range(self._max_optimization_epochs):
            epoch_loss = 0.0
            for batch_inputs, batch_targets in dataloader:
                self.optimizer.zero_grad()
                loss = self._model.compute_loss(batch_inputs, batch_targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)

            if self._verbose:
                print(f'{self._name} - Epoch {epoch + 1}, Loss: {epoch_loss}')

            if self._callback:
                self._callback(epoch, epoch_loss)

            if abs(prev_loss - epoch_loss) < self._tolerance:
                if self._verbose:
                    print(
                        f"{self._name} - Early stopping due to tolerance level reached.")
                break

            prev_loss = epoch_loss
