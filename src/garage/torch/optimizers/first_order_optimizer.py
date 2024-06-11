import os
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

    def __init__(
        self,
        model,
        optimizer_class=torch.optim.Adam,
        learning_rate=1e-3,
        max_optimization_epochs=1000,
        tolerance=1e-6,
        batch_size=32,
        callback=None,
        verbose=False,
        name='FirstOrderOptimizer',
        load_state=False,
        state_dir=None,
    ):
        self._model = model
        self._optimizer_class = optimizer_class
        self._learning_rate = learning_rate
        self._max_optimization_epochs = max_optimization_epochs
        self._tolerance = tolerance
        self._batch_size = batch_size
        self._callback = callback
        self._verbose = verbose
        self._name = name
        self.optimizer = self._optimizer_class(model.parameters(),
                                                lr=self._learning_rate)
        if state_dir is None:
            self.state_dir = f"saved_models/rl_2_optim_{name}.pth"
        else:
            self.state_dir = f"{state_dir}/{name}.path"
        self._loss_function = None
        if load_state:
            self.load_optimizer_state()
            pass

    def update_opt(self, loss_function):
        """Sets up the optimizer and loss function based on provided arguments.

        Args:
            loss_function (callable): The loss function to be used for optimization.
        """
        self._loss_function = loss_function

    def optimize(self, inputs, targets=None):
        """Executes the optimization process.

        Args:
            inputs (torch.Tensor or list of torch.Tensor): The input data.
            targets (torch.Tensor, optional): The target data.
        """
        # Ensure inputs is a list of tensors
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]  # Convert single tensor to a list of one tensor

        # Create the dataset depending on whether targets are provided
        if targets is not None:
            dataset = TensorDataset(*inputs, targets)
        else:
            dataset = TensorDataset(*inputs)

        # Create a DataLoader to handle batching
        dataloader = DataLoader(dataset, batch_size=self._batch_size,
                                shuffle=True)

        prev_loss = float('inf')
        for epoch in range(self._max_optimization_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                self.optimizer.zero_grad()

                # Proper unpacking: Check if targets are included and unpack accordingly
                if targets is not None:
                    *batch_inputs, batch_targets = batch
                    loss = self._loss_function(*batch_inputs,
                                                    batch_targets)
                else:
                    batch_inputs = batch if len(inputs) > 1 else (batch[0],)
                    loss = self._loss_function(*batch_inputs)
                loss = loss[0] if isinstance(loss, (list, tuple)) else loss

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)

            if self._verbose:
                print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')

            if self._callback:
                self._callback(epoch, epoch_loss)

            if abs(prev_loss - epoch_loss) < self._tolerance:
                if self._verbose:
                    print("Early stopping due to tolerance level reached.")
                break

            prev_loss = epoch_loss

    def save_optimizer_state(self):
        params = self.optimizer.state_dict()
        os.makedirs(os.path.dirname(self.state_dir), exist_ok=True)
        torch.save(params, self.state_dir)

    def load_optimizer_state(self):
        state_dict = torch.load(self.state_dir)
        # to guarantee proper dtypes, we first must move
        # model to optim dtype and then back to where it was
        model_dtype = next(self._model.parameters()).dtype
        for p in state_dict["state"][0].values():
            if isinstance(p, torch.Tensor):
                optim_dtype = p.dtype
                break
        self._model.to(dtype=optim_dtype)
        self.optimizer.load_state_dict(state_dict)
        self._model.to(dtype=model_dtype)


