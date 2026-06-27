import torch
import torch.nn as nn

class SingleFrequencyModel(nn.Module):
    """
    Neural Network model for fundamental frequency (1H) positioning control.
    """
    def __init__(self, in_dim=4, layer_dims=[60], out_dim=2, dropout=0.1):
        super(SingleFrequencyModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_dims = layer_dims
        
        # Define dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Build layer stack
        self.layers = nn.ModuleList()
        current_dim = in_dim
        
        # Add hidden layers
        for hidden_dim in self.layer_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
            
        # Add output layer
        self.layers.append(nn.Linear(current_dim, out_dim))

    def forward(self, x):
        # Extract real and imaginary components from the 2-column input [Real, Imag]
        real_part = x[:, 0].unsqueeze(1)
        imag_part = x[:, 1].unsqueeze(1)
        
        # Expand input to 4 columns to accommodate magnitude and phase features
        x = torch.cat((real_part, imag_part, real_part, imag_part), dim=1)
        
        # Calculate magnitude: sqrt(real^2 + imag^2)
        x[:, 2] = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
        
        # Calculate phase normalized in radians: atan2(imag, real) / pi
        x[:, 3] = torch.atan2(x[:, 1], x[:, 0]) / torch.pi

        # Apply dropout
        x = self.dropout(x)
        
        # Pass through hidden layers with Tanh activation
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
            
        # Apply output layer with Sigmoid activation
        out = torch.sigmoid(self.layers[-1](x))

        # Denormalization of outputs:
        # X1 motor position desnormalization
        denorm_x1 = out[:, 0] * 4000
        # Y1 motor position desnormalization
        denorm_y1 = (out[:, 1] * 2940) + 6000
        
        # Stack the denormalized outputs into shape (N, 2)
        out = torch.stack((denorm_x1, denorm_y1), dim=1)
        return out


class DualFrequencyModel(nn.Module):
    """
    Neural Network model for fundamental and second harmonic frequency (2H) positioning control.
    """
    def __init__(self, in_dim=8, layer_dims=[60], out_dim=4, dropout=0.1):
        super(DualFrequencyModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_dims = layer_dims
        
        # Define dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Build layer stack
        self.layers = nn.ModuleList()
        current_dim = in_dim
        
        # Add hidden layers
        for hidden_dim in self.layer_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
            
        # Add output layer
        self.layers.append(nn.Linear(current_dim, out_dim))

    def forward(self, x):
        # Extract components from the 4-column input [Real_1H, Imag_1H, Real_2H, Imag_2H]
        real_1h = x[:, 0].unsqueeze(1)
        imag_1h = x[:, 1].unsqueeze(1)
        real_2h = x[:, 2].unsqueeze(1)
        imag_2h = x[:, 3].unsqueeze(1)

        # Expand input to 8 columns to accommodate magnitude and phase features for both frequencies
        x = torch.cat((real_1h, imag_1h, real_2h, imag_2h, real_1h, imag_1h, real_2h, imag_2h), dim=1)

        # 1H (Fundamental Harmonic) Feature Extraction
        # Calculate magnitude
        x[:, 4] = torch.sqrt(x[:, 0]**2 + x[:, 1]**2) 
        # Calculate normalized phase (radians)
        x[:, 5] = torch.atan2(x[:, 1], x[:, 0]) / torch.pi 

        # 2H (Second Harmonic) Feature Extraction
        # Calculate magnitude
        x[:, 6] = torch.sqrt(x[:, 2]**2 + x[:, 3]**2)
        # Calculate normalized phase (radians)
        x[:, 7] = torch.atan2(x[:, 3], x[:, 2]) / torch.pi

        # Apply dropout
        x = self.dropout(x)
        
        # Pass through hidden layers with Tanh activation
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
            
        # Apply output layer with Sigmoid activation
        out = torch.sigmoid(self.layers[-1](x))

        # Denormalization of outputs:
        # X1 motor position desnormalization
        denorm_x1 = out[:, 0] * 4000
        # Y1 motor position desnormalization
        denorm_y1 = (out[:, 1] * 2940) + 6000
        # X2-X1 relative motor position desnormalization
        denorm_x2_minus_x1 = (out[:, 2] * 2300) + 700
        # Y2 motor position desnormalization
        denorm_y2 = (out[:, 3] * 2960) + 6000
        
        # Stack the denormalized outputs into shape (N, 4)
        out = torch.stack((denorm_x1, denorm_y1, denorm_x2_minus_x1, denorm_y2), dim=1)
        return out
