import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################################
# Activation Function
##############################################################################
import math
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
    # If x is [batch_size, input_size], add the sequence dimension:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # -> [batch_size, 1, input_size]
        out, _ = self.lstm(x)   # Now out is [batch_size, 1, hidden_size]
        out = self.fc(out[:, -1, :])  # [batch_size, hidden_size] -> [batch_size, output_size]
        return out

    # def forward(self, x):
    #     out, _ = self.lstm(x)
    #     out = self.fc(out[:, -1, :])
    #     return out
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
    # If x is [batch_size, input_size], add the sequence dimension:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # -> [batch_size, 1, input_size]
        out, _ = self.gru(x)   # Now out is [batch_size, 1, hidden_size]
        out = self.fc(out[:, -1, :])  # [batch_size, hidden_size] -> [batch_size, output_size]
        return out

    # def forward(self, x):
    #     out, _ = self.lstm(x)
    #     out = self.fc(out[:, -1, :])
    #     return out



class ReciprocalActivation(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (x + self.epsilon)

import torch
import torch.nn as nn

class ReciprocalActivation(nn.Module):
    """Reciprocal activation: out = 1 / (x + eps)."""
    def __init__(self, eps=1e-20):
        super().__init__()
        self.eps = eps

    def forward(self, input):
        return 1.0 / (input + self.eps)


# class CauchyNet0(nn.Module):
#     """CauchyNet returning (y_real, y_imag) for scaled outputs, with fixed xi."""
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size

#         # 1) Trainable lambda_
#         self.lambda_ = nn.Parameter(
#             torch.normal(mean=0.0, std=1.0,
#                          size=(hidden_size, output_size),
#                          dtype=torch.cfloat)
#         )

#         # 2) Generate elliptical initialization for xi, but fix (register_buffer).
#         angles = 2.0 * torch.pi * torch.rand(hidden_size)
#         real_part      = 1.5 * torch.cos(angles)
#         imaginary_part = 0.5 * torch.sin(angles)
#         xi_init        = torch.complex(real_part, imaginary_part)

#         # Instead of nn.Parameter, we do register_buffer:
#         self.register_buffer("xi_fixed", xi_init)

#         # 3) Reciprocal Activation
#         self.activation = ReciprocalActivation(eps=1e-20)

#     def forward(self, x):
#         # Ensure x shape is (batch_size, 1)
#         if x.dim() == 1:
#             x = x.unsqueeze(-1)

#         batch_size, _ = x.size()

#         # Convert x to complex => shape (batch_size, 1)
#         x_c = torch.complex(x, torch.zeros_like(x))  # (batch_size, 1)

#         # Expand xi => shape (batch_size, hidden_size)
#         xi_expanded = self.xi_fixed.unsqueeze(0).expand(batch_size, -1)

#         # Expand x => shape (batch_size, hidden_size)
#         x_expanded  = x_c.expand(batch_size, self.hidden_size)

#         # Reciprocal: 1 / (xi - x)
#         activated = self.activation(xi_expanded - x_expanded)  # (batch_size, hidden_size)

#         # (batch_size, hidden_size) x (hidden_size, output_size) => (batch_size, output_size)
#         y_complex = torch.matmul(activated, self.lambda_) / self.hidden_size

#         # Return real & imaginary parts separately
#         return y_complex.real, y_complex.imag
class CauchyNet0(nn.Module):
    """
    Only lambda_ is trainable; xi is fixed (angles).
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Trainable lambda
        self.lambda_ = nn.Parameter(
            torch.randn(hidden_size, output_size, dtype=torch.cfloat)
        )

        # Fix xi in a buffer (not trained)
        angles = 2*math.pi*torch.rand(hidden_size)
        real_part = 3.0*torch.cos(angles)
        imag_part = 0.5*torch.sin(angles)

        # register_buffer(...) => NOT a Parameter => not trainable
        self.register_buffer("xi_fixed", torch.complex(real_part, imag_part))

        self.activation = ReciprocalActivation()

    def forward(self, x: torch.Tensor):
        # x shape [B,1]
        B = x.shape[0]
        x_flat = x.view(B)  # [B]
        x_c    = torch.complex(x_flat, torch.zeros_like(x_flat))  # [B]
        x_expanded  = x_c.unsqueeze(1).expand(B, self.hidden_size)     # => [B, hidden_size]
        xi_expanded = self.xi_fixed.unsqueeze(0).expand(B, self.hidden_size)

        activated = self.activation(xi_expanded - x_expanded)  # [B, hidden_size]
        out_c     = torch.matmul(activated, self.lambda_) / self.hidden_size  # => [B, out_features]
        return out_c.real, out_c.imag        
class ReciprocalActivation(nn.Module):
    """Reciprocal activation: out = 1 / (x + epsilon)."""
    def __init__(self, epsilon=1e-10):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return 1.0 / (x + self.epsilon)

class ReciprocalActivation(nn.Module):
    """Reciprocal activation: out = 1 / (x + epsilon)."""
    def __init__(self, eps=1e-20):
        super().__init__()
        self.eps = eps

    def forward(self, input):
        return 1.0 / (input + self.eps)
# CauchyNet1 is better
class CauchyNet1(nn.Module):
    """CauchyNet returning (y_real, y_imag) for scaled outputs."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 1) lambda_ remains normal
        self.lambda_ = nn.Parameter(
            torch.normal(mean=0.0, std=1.0, size=(hidden_size, output_size), dtype=torch.cfloat)
        )

        # 2) Elliptical initialization for xi
        # Generate random angles in [0, 2*pi]
        angles = 2.0 * torch.pi * torch.rand(hidden_size)

        # Real part radius = 1.5, Imag part radius = 0.5
        real_part      = 1.5 * torch.cos(angles)
        imaginary_part = 0.5 * torch.sin(angles)

        # Combine into a complex tensor
        xi_init = torch.complex(real_part, imaginary_part)
        # Register as a trainable parameter
        self.xi = nn.Parameter(xi_init)

        # Reciprocal Activation
        self.activation = ReciprocalActivation()

    def forward(self, x):
        # If x has shape (batch,) => unsqueeze to (batch,1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        batch_size, _ = x.size()

        # Convert x to complex => shape (batch_size,1)
        x_c = torch.complex(x, torch.zeros_like(x))

        # Expand xi => shape (batch_size, hidden_size)
        xi_expanded = self.xi.unsqueeze(0).expand(batch_size, -1)

        # Expand x => shape (batch_size, hidden_size)
        x_expanded  = x_c.expand(batch_size, self.hidden_size)

        # activated => reciprocal( xi - x )
        activated = self.activation(xi_expanded - x_expanded)  # (batch_size, hidden_size)

        # Multiply by lambda_ => shape => (batch_size, hidden_size) x (hidden_size, output_size) => (batch_size, output_size)
        # Then optionally scale by 1/hidden_size (as in your code)
        y_complex = torch.matmul(activated, self.lambda_) / self.hidden_size

        # Return real & imag, each shape => (batch_size, output_size)
        return y_complex.real, y_complex.imag
        
##############################################################################
# Models
##############################################################################
class CauchyNet(nn.Module):
    """CauchyNet returning (y_real, y_imag) for scaled outputs."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lambda_ = nn.Parameter(torch.normal(mean=0.0, std=1, size=(hidden_size, output_size), dtype=torch.cfloat))
        self.xi = nn.Parameter(torch.normal(mean=0.0, std=1, size=(hidden_size,), dtype=torch.cfloat))
        self.activation = ReciprocalActivation()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        batch_size, _ = x.size()
        x_c = torch.complex(x, torch.zeros_like(x))
        xi_expanded = self.xi.unsqueeze(0).expand(batch_size, -1)
        x_expanded = x_c.expand(batch_size, self.hidden_size)
        activated = self.activation(xi_expanded - x_expanded)
        y_complex = torch.matmul(activated, self.lambda_) / self.hidden_size
        return y_complex.real, y_complex.imag

class SIREN(nn.Module):
    """SIREN: Implicit Neural Representations with Periodic Activation Functions."""
    def __init__(self, input_size, hidden_size, output_size, w0_initial=30.0, w0_hidden=30.0):
        super().__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)
        self.w0_initial = w0_initial
        self.w0_hidden = w0_hidden
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.uniform_(self.linear_in.weight, -1.0 / self.linear_in.in_features, 1.0 / self.linear_in.in_features)
            nn.init.zeros_(self.linear_in.bias)
            self.linear_in.weight *= self.w0_initial
            bound = (6.0 / self.linear_out.in_features) ** 0.5
            nn.init.uniform_(self.linear_out.weight, -bound / self.w0_hidden, bound / self.w0_hidden)
            nn.init.zeros_(self.linear_out.bias)

    def sine(self, x, w0):
        return torch.sin(w0 * x)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        h = self.sine(self.linear_in(x), w0=1.0)
        out = self.linear_out(h)
        return out


class RBFNetwork(nn.Module):
    """Radial Basis Function Network."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(hidden_size, input_size))
        self.log_sigmas = nn.Parameter(torch.zeros(hidden_size))
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x_expanded = x.unsqueeze(1)
        centers_expanded = self.centers.unsqueeze(0)
        dist_sq = (x_expanded - centers_expanded).pow(2).sum(dim=2)
        sigma = torch.exp(self.log_sigmas).unsqueeze(0)
        rbf_activations = torch.exp(-dist_sq / (2 * sigma.pow(2)))
        out = self.linear(rbf_activations)
        return out


class FNN_ReLU(nn.Module):
    """Feedforward Neural Network with ReLU Activation."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return out


class FNN_Sigmoid(nn.Module):
    """Feedforward Neural Network with Sigmoid Activation."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return out


##############################################################################
# 2) Minimal Transformer & Informer with batch_first=True
##############################################################################
# class MinimalTransformer(nn.Module):
#     """
#     A minimal single-step Transformer for function approximation (scalar input->output).
#     """
#     def __init__(self, input_size, hidden_size, output_size, nhead=4, num_layers=1):
#         super().__init__()
#         self.embedding = nn.Linear(input_size, hidden_size)
#         # batch_first=True to avoid the nested_tensor warning
#         encoder_layer  = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
#         self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.fc_out    = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # x: (batch, input_dim=1)
#         # We'll treat each sample as a "sequence of length=1"
#         emb = self.embedding(x)                          # (batch, hidden_size)
#         emb = emb.unsqueeze(1)                           # (batch, seq=1, hidden_size)
#         enc = self.encoder(emb)                          # (batch, seq=1, hidden_size)
#         enc = enc.squeeze(1)                             # (batch, hidden_size)
#         out = self.fc_out(enc)                           # (batch, output_size)
#         return out

# class MinimalInformer(nn.Module):
#     """
#     A toy 'Informer-like' model for demonstration with batch_first=True.
#     """
#     def __init__(self, input_size, hidden_size, output_size, nhead=4, num_layers=1):
#         super().__init__()
#         self.embedding = nn.Linear(input_size, hidden_size)
#         self.gate      = nn.Linear(hidden_size, hidden_size)
#         encoder_layer  = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
#         self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.fc_out    = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         emb  = self.embedding(x)             # (batch, hidden_size)
#         gate = torch.sigmoid(self.gate(emb)) * emb
#         gate = gate.unsqueeze(1)             # (batch, seq=1, hidden_size)
#         enc  = self.encoder(gate)            # (batch, seq=1, hidden_size)
#         enc  = enc.squeeze(1)                # (batch, hidden_size)
#         out  = self.fc_out(enc)              # (batch, output_size)
#         return out
import torch
import torch.nn as nn

class MinimalTransformer(nn.Module):
    """
    Optimized minimal Transformer for scalar input-output mappings.
    """
    def __init__(self, input_size, hidden_size, output_size, nhead=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, hidden_size))  # Positional encoding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 2, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Ensure x has the shape (batch_size, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Embedding and positional encoding
        x = self.embedding(x)                          # Shape: (batch_size, hidden_size)
        x = x.unsqueeze(1) + self.positional_encoding  # Shape: (batch_size, seq_len=1, hidden_size)
        x = self.dropout(x)                            # Apply dropout

        # Transformer encoding
        x = self.encoder(x)                            # Shape: (batch_size, seq_len=1, hidden_size)
        x = x.squeeze(1)                               # Remove sequence dimension: (batch_size, hidden_size)

        # Fully connected output layer
        out = self.fc_out(x)                           # Shape: (batch_size, output_size)
        return out

class MinimalInformer(nn.Module):
    """
    Optimized Informer-like model for scalar input-output mappings.
    """
    def __init__(self, input_size, hidden_size, output_size, nhead=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 2, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Ensure x has the shape (batch_size, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Embedding and gating
        emb = self.embedding(x)                        # Shape: (batch_size, hidden_size)
        gate = torch.sigmoid(self.gate(emb)) * emb     # Apply gating
        gate = gate.unsqueeze(1)                       # Shape: (batch_size, seq_len=1, hidden_size)
        gate = self.dropout(gate)                      # Apply dropout

        # Transformer encoding
        enc = self.encoder(gate)                       # Shape: (batch_size, seq_len=1, hidden_size)
        enc = enc.squeeze(1)                           # Remove sequence dimension: (batch_size, hidden_size)

        # Fully connected output layer
        out = self.fc_out(enc)                         # Shape: (batch_size, output_size)
        return out

##############################################################################
# N-BEATS Blocks
##############################################################################
class NBeatsBlock(nn.Module):
    """
    A single N-BEATS block:
      - Fully-connected layers (or minimal subset) with ReLU
      - Output layer for forecast
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Optionally add more layers here if you like:
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x shape: (batch_size, input_size)
        Returns: shape (batch_size, output_size)
        """
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        out = self.fc_out(x)
        return out


class NBeats(nn.Module):
    """
    Minimal N-BEATS style model:
      - Stacks multiple NBeatsBlock blocks
      - Uses the "residual" trick: each block refines the forecast
    """
    def __init__(self, input_size, hidden_size, output_size, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, hidden_size, output_size)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        x shape: (batch_size, input_size)
        We'll do the standard N-BEATS approach: 
          residual <- x
          For each block:
            out = block(residual)
            residual = residual - out
          Return final out from the last block
        """
        # If x is 1D => unsqueeze
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # x shape: (batch_size, input_size)
        residual = x
        final_out = None
        for block in self.blocks:
            out = block(residual)
            residual = residual - out
            final_out = out  # keep track of the last block's output

        return final_out