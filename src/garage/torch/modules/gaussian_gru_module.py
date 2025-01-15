import torch
import torch.nn as nn

class GaussianGRUModule(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=32,
        learn_std=True,
        init_std=1.0,
        min_std=None,
        max_std=None,
        std_share_network=False,
        layer_normalization=False,
        output_nonlinearity=None
    ):
        super(GaussianGRUModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.learn_std = learn_std
        self.std_share_network = std_share_network
        self.layer_normalization = layer_normalization
        self.output_nonlinearity = output_nonlinearity
        self._init_std_param = torch.log(torch.tensor(init_std))
        if min_std is not None and max_std is not None:
            self.log_min_std = torch.log(torch.tensor(min_std))
            self.log_max_std = torch.log(torch.tensor(max_std))
        else:
            self.log_min_std, self.log_max_std = None, None

        # Define GRU layers
        self.gru_mean = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        for name, param in self.gru_mean.named_parameters():
            if 'weight' in name:
                # Apply Xavier (Glorot) Uniform Initialization to all weight matrices
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Initialize biases to zero (optional, can be constant or other strategies too)
                torch.nn.init.zeros_(param)
        if std_share_network:
            self.gru_std = None  # Share the same GRU for mean and std
            # Define output layers for mean and std
            self.fc_mean_std = nn.Linear(hidden_dim, output_dim*2)
            torch.nn.init.xavier_uniform_(self.fc_mean_std.weight)
            self.fc_mean = None
            self.fc_std = None
        else:
            # Not used and probably incorrect
            self.gru_std = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
            for name, param in self.gru_std.named_parameters():
                if 'weight' in name:
                    # Apply Xavier (Glorot) Uniform Initialization to all weight matrices
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    # Initialize biases to zero (optional, can be constant or other strategies too)
                    torch.nn.init.zeros_(param)
            self.fc_mean = nn.Linear(hidden_dim, output_dim)
            self.fc_std = nn.Linear(hidden_dim, output_dim)
            torch.nn.init.xavier_uniform_(self.fc_mean.weight)
            torch.nn.init.xavier_uniform_(self.fc_std.weight)

            if self.learn_std:
                self.fc_std = nn.Linear(hidden_dim, output_dim)
                torch.nn.init.xavier_uniform_(self.fc_std.weight)
                self.std_param = nn.Parameter(torch.full((output_dim,), init_std))

            else:
                self.register_buffer('std_param', torch.full((output_dim,), init_std))

        if layer_normalization:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None

    def forward(self, x, hidden):
        # Process through GRU
        output_mean, hidden = self.gru_mean(x, hidden)
        if self.layer_normalization:
            output_mean = self.layer_norm(output_mean)
        if self.std_share_network:
            # means are first four elements and std four last elements
            mean_std = self.output_nonlinearity(self.fc_mean_std(output_mean))
            mean = mean_std[..., :self.output_dim]
            log_std = mean_std[..., self.output_dim: ]
        else:
            mean = self.output_nonlinearity(self.fc_mean(output_mean))
            output_std, self.hidden_std = self.gru_std(x, self.hidden_std)
            if self.layer_normalization:
                output_std = self.layer_norm(output_std)
            log_std = self.output_nonlinearity(self.fc_std(output_std))

        if self.log_min_std  and self.log_max_std:
            def _softclip(x, x_min, x_max, alpha=2):
                y_scale = (x_max - x_min) / 2
                y_offset = (x_max + x_min) / 2
                x_scale = (2 * alpha) / (x_max - x_min)
                x_offset = (x_max + x_min) / 2
                return (torch.tanh((x - x_offset) * x_scale) * y_scale) + y_offset
            log_std = _softclip(log_std, self.log_min_std, self.log_max_std)
        std = torch.exp(log_std)
        distribution = torch.distributions.Normal(mean, std)

        return distribution, hidden
