import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size

        # Key, Query, and Value matrices
        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # Apply linear transformations
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.hidden_size ** 0.5
        weights = F.softmax(scores, dim=-1)

        # Apply attention weights
        attention = torch.matmul(weights, value)
        return attention

class multiOutputNet(nn.Module):
    def __init__(self, size_in: int, hidden_sizes_shared: list, hidden_sizes_a: list, hidden_sizes_b: list, size_out1: int, size_out2: int, dropout_p: float = 0.5):
        super(multiOutputNet, self).__init__()
        self.size_in = size_in
        self.hidden_sizes_shared = hidden_sizes_shared
        self.hidden_sizes_a = hidden_sizes_a
        self.hidden_sizes_b = hidden_sizes_b
        self.size_out1 = size_out1
        self.size_out2 = size_out2
        self.dropout_p = dropout_p

        # Shared layers
        self.shared_layers = nn.ModuleList()
        prev_size = self.size_in
        for size in self.hidden_sizes_shared:
            self.shared_layers.append(nn.Sequential(
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p)
            ))
            prev_size = size

        self.attention = SelfAttention(self.hidden_sizes_shared[-1])
        # Layers for output 1
        self.hidden_layers_a = self._make_layers(self.hidden_sizes_a)

        # Layers for output 2
        self.hidden_layers_b = self._make_layers(self.hidden_sizes_b)

        # Output layers
        self.output_layer1 = nn.Linear(self.hidden_sizes_a[-1], self.size_out1)
        self.output_layer2 = nn.Linear(self.hidden_sizes_b[-1], self.size_out2)

    def _make_layers(self, layer_sizes):
        layers = nn.ModuleList()
        prev_size = self.hidden_sizes_shared[-1]
        for size in layer_sizes:
            layers.append(nn.Sequential(
                nn.Linear(prev_size, size),
                # nn.BatchNorm1d(size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p)
            ))
            prev_size = size
        return layers

    def forward(self, x):
        # Shared layers
        for layer in self.shared_layers:
            x = layer(x)
        # x = self.attention(x)
        # Path for output 1
        x_a = x
        x_b = x
        for layer in self.hidden_layers_a:
            x_a = layer(x_a)

        # Path for output 2
        for layer in self.hidden_layers_b:
            x_b = layer(x_b)

        # # Weighted sum of outputs from paths A and B
        # combined = self.weight_a * x_a + self.weight_b * x_b

        # # Feeding the combined output back into the shared layers
        # for layer in self.shared_layers:
        #     combined = layer(combined)

        # Final outputs
        out1 = self.output_layer1(x_a)
        out2 = self.output_layer2(x_b)

        return out1, out2
    
class ModalityTypeClassifier(nn.Module):
    """docstring for ModalityTypeClassifier."""
    def __init__(self, size_in, num_hidden, hidden_size, size_out, dropout_p = 0.5):
        super(ModalityTypeClassifier, self).__init__()
        self.size_in = size_in
        self.num_shared_layers = num_hidden
        self.hidden_size = hidden_size
        self.size_out = size_out
        self.dropout_p = dropout_p
        
        self.input = nn.Sequential(
            nn.Linear(self.size_in, self.hidden_size),
            nn.ReLU()
        )
        # self.attention = SelfAttention(self.hidden_size)
        self.shared_layers = nn.ModuleList()
        for _ in range(0, self.num_shared_layers):
            self.shared_layers.append(nn.Sequential(
                # nn.BatchNorm1d(self.hidden_layer_size),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p)
            ))
        self.output_layer = nn.Linear(self.hidden_size, self.size_out)
        
    def forward(self, x):
        x = self.input(x)
        # x = self.attention(x) # attention mechanism not used
        
        # Shared layers
        for layer in self.shared_layers:
            # layer_out = layer(x)
            x = layer(x)
            # x = self.attention(x) # residual connections with attention mechanism
            
        x = self.output_layer(x)
        
        return x
