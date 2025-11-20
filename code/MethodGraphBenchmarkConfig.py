from transformers.configuration_utils import PretrainedConfig

class GCNConfig(PretrainedConfig):
    def __init__(
        self,
        x_size=3000,
        y_size=7,
        hidden_size=64,
        dropout=0.3,
        num_layers=2,
        lr=0.01,
        weight_decay=5e-4,
        max_epoch=500,
        **kwargs
    ):
        super(GCNConfig, self).__init__(**kwargs)
        self.x_size = x_size                # Input feature size
        self.y_size = y_size                # Number of output classes
        self.hidden_size = hidden_size      # Hidden layer size
        self.dropout = dropout              # Dropout probability
        self.num_layers = num_layers        # Number of GCN layers (if using more than 2)
        self.lr = lr                        # Learning rate
        self.weight_decay = weight_decay    # Weight decay (L2 regularization)
        self.max_epoch = max_epoch          # Max training epochs