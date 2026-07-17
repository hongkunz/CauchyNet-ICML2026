import math
import torch
import torch.nn as nn


##############################################################################
# 0) Common Helper: Flatten input to shape [B]
##############################################################################
def _flatten_input(x: torch.Tensor) -> torch.Tensor:
    """
    Ensures x becomes shape [B].
    Accepts [B], [B,1], or [B,1,1].
    Raises ValueError otherwise.
    """
    # If 3D => could be [B,1,1]. Squeeze down to [B].
    if x.dim() == 3:
        # e.g. [B,1,1]
        if x.size(1) == 1 and x.size(2) == 1:
            x = x.squeeze(-1).squeeze(-1)  # => [B]
        else:
            raise ValueError(f"Unexpected 3D shape {tuple(x.shape)}")
    elif x.dim() == 2:
        # e.g. [B,1]
        if x.size(1) != 1:
            raise ValueError(f"Expected [B,1], got {tuple(x.shape)}")
        x = x.squeeze(-1)  # => [B]
    elif x.dim() == 1:
        # Already [B]
        pass
    else:
        raise ValueError(f"Input x must be [B], [B,1], or [B,1,1], got {tuple(x.shape)}")

    return x

##############################################################################
# 1) Reciprocal Activation
##############################################################################
class ReciprocalActivation(nn.Module):
    """
    Reciprocal activation: out = 1 / (x + epsilon).
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (x + self.epsilon)


##############################################################################
# 2) CauchyNet0: Trainable lambda_, fixed xi
##############################################################################
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

        # Fix xi in a buffer => not trainable
        angles = 2*math.pi*torch.rand(hidden_size)
        real_part = 3.0*torch.cos(angles)
        imag_part = 0.5*torch.sin(angles)
        self.register_buffer("xi_fixed", torch.complex(real_part, imag_part))

        self.activation = ReciprocalActivation()

    def forward(self, x: torch.Tensor):
        # Unify shape => [B]
        x_flat = _flatten_input(x)  # => [B]
        B = x_flat.shape[0]

        # Convert real -> complex => shape [B]
        x_c = torch.complex(x_flat, torch.zeros_like(x_flat))

        # Expand => [B, hidden_size]
        x_expanded  = x_c.unsqueeze(1).expand(B, self.hidden_size)
        xi_expanded = self.xi_fixed.unsqueeze(0).expand(B, self.hidden_size)

        # reciprocal( xi - x )
        activated = self.activation(xi_expanded - x_expanded)  # [B, hidden_size]

        # Multiply by trainable lambda_
        out_c = torch.matmul(activated, self.lambda_) / self.hidden_size

        return out_c.real, out_c.imag


##############################################################################
# 2) CauchyNet1: Trainable lambda_, trainable elliptical xi
##############################################################################
class CauchyNet1(nn.Module):
    """
    Trainable lambda_ and xi, each xi element is elliptical init:
    Real radius=1.5, Imag radius=0.5.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Trainable lambda (complex)
        self.lambda_ = nn.Parameter(
            torch.normal(mean=0.0, std=1.0,
                         size=(hidden_size, output_size), dtype=torch.cfloat)
        )

        # Elliptical initialization for xi
        angles = 2.0 * torch.pi * torch.rand(hidden_size)
        real_part = 1.5 * torch.cos(angles)
        imaginary_part = 0.5 * torch.sin(angles)
        xi_init = torch.complex(real_part, imaginary_part)
        self.xi = nn.Parameter(xi_init)

        self.activation = ReciprocalActivation()

    def forward(self, x):
        # Flatten => [B]
        x_flat = _flatten_input(x)
        B = x_flat.shape[0]

        # x => complex => [B]
        x_c = torch.complex(x_flat, torch.zeros_like(x_flat))

        # Expand => [B, hidden_size]
        x_expanded  = x_c.unsqueeze(1).expand(B, self.hidden_size)
        xi_expanded = self.xi.unsqueeze(0).expand(B, self.hidden_size)

        # Activation => 1/(xi - x)
        activated = self.activation(xi_expanded - x_expanded)

        y_complex = torch.matmul(activated, self.lambda_) / self.hidden_size
        return y_complex.real, y_complex.imag


# 3) CauchyNet1: Trainable lambda_, trainable elliptical xi
##############################################################################
class CauchyNet1_NoImagPenalty(nn.Module):
    """
    Trainable lambda_ and xi, each xi element is elliptical init:
    Real radius=1.5, Imag radius=0.5.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Trainable lambda (complex)
        self.lambda_ = nn.Parameter(
            torch.normal(mean=0.0, std=1.0,
                         size=(hidden_size, output_size), dtype=torch.cfloat)
        )

        # Elliptical initialization for xi
        angles = 2.0 * torch.pi * torch.rand(hidden_size)
        real_part = 1.5 * torch.cos(angles)
        imaginary_part = 0.5 * torch.sin(angles)
        xi_init = torch.complex(real_part, imaginary_part)
        self.xi = nn.Parameter(xi_init)

        self.activation = ReciprocalActivation()

    def forward(self, x):
        # Flatten => [B]
        x_flat = _flatten_input(x)
        B = x_flat.shape[0]

        # x => complex => [B]
        x_c = torch.complex(x_flat, torch.zeros_like(x_flat))

        # Expand => [B, hidden_size]
        x_expanded  = x_c.unsqueeze(1).expand(B, self.hidden_size)
        xi_expanded = self.xi.unsqueeze(0).expand(B, self.hidden_size)

        # Activation => 1/(xi - x)
        activated = self.activation(xi_expanded - x_expanded)

        y_complex = torch.matmul(activated, self.lambda_) / self.hidden_size
        return y_complex.real

##############################################################################
# 1) CauchyNet: Trainable lambda_, trainable xi (random normal init)
##############################################################################
class CauchyNet(nn.Module):
    """
    Original version with random normal initialization for xi.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lambda_ = nn.Parameter(
            torch.normal(mean=0.0, std=1.0,
                         size=(hidden_size, output_size), dtype=torch.cfloat)
        )
        self.xi = nn.Parameter(
            torch.normal(mean=0.0, std=1.0,
                         size=(hidden_size,), dtype=torch.cfloat)
        )
        self.activation = ReciprocalActivation()
    def forward(self, x):
        x_flat = _flatten_input(x)  # => [B]
        B = x_flat.shape[0]

        x_c = torch.complex(x_flat, torch.zeros_like(x_flat))  # => [B]
        x_expanded  = x_c.unsqueeze(1).expand(B, self.hidden_size)
        xi_expanded = self.xi.unsqueeze(0).expand(B, self.hidden_size)

        activated = self.activation(xi_expanded - x_expanded)
        y_complex = torch.matmul(activated, self.lambda_) / self.hidden_size
      
        return y_complex.real, y_complex.imag
        
# class CauchyNet(nn.Module):
#     """
#     Original version with random normal initialization for xi.
#     """
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size

#         self.lambda_ = nn.Parameter(
#             torch.normal(mean=0.0, std=1.0,
#                          size=(hidden_size, output_size), dtype=torch.cfloat)
#         )
#         self.xi = nn.Parameter(
#             torch.normal(mean=0.0, std=1.0,
#                          size=(hidden_size,), dtype=torch.cfloat)
#         )
#         self.activation = ReciprocalActivation()

#     def forward(self, x):
#         x_flat = _flatten_input(x)  # => [B]
#         B = x_flat.shape[0]

#         x_c = torch.complex(x_flat, torch.zeros_like(x_flat))  # => [B]
#         x_expanded  = x_c.unsqueeze(1).expand(B, self.hidden_size)
#         xi_expanded = self.xi.unsqueeze(0).expand(B, self.hidden_size)

#         activated = self.activation(xi_expanded - x_expanded)
#         y_complex = torch.matmul(activated, self.lambda_) / self.hidden_size
#         return y_complex.real, y_complex.imag


##############################################################################
# 5) CauchyNet_RealActivation
##############################################################################
class CauchyNet_RealActivation(nn.Module):
    """
    A 'CauchyNet' variant that uses a real ReLU activation on
    real differences instead of a complex reciprocal.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Real trainable parameters for location
        self.real_part = nn.Parameter(torch.randn(hidden_size))
        # Real weights for output
        self.lambda_real = nn.Parameter(torch.randn(hidden_size, output_size))
        self.lambda_imag = nn.Parameter(torch.randn(hidden_size, output_size))
        # Activation
        self.activation = nn.ReLU()

    def forward(self, x):
        # Flatten => [B]
        x_flat = _flatten_input(x)
        B = x_flat.shape[0]
        # real_diff => shape [B, hidden_size]
        # expand 'self.real_part' => [hidden_size] to => [B, hidden_size]
        real_part_expanded = self.real_part.unsqueeze(0).expand(B, self.hidden_size)
        real_diff = x_flat.unsqueeze(-1).expand(B, self.hidden_size) - real_part_expanded
        activated = self.activation(real_diff)  # => [B, hidden_size]
        y_real = torch.matmul(activated, self.lambda_real) / self.hidden_size
        y_imag = torch.matmul(activated, self.lambda_imag) / self.hidden_size
        return y_real, y_imag


##############################################################################
# 6) CauchyNet_NoImagPenalty
##############################################################################
class CauchyNet_NoImagPenalty(nn.Module):
    """
    Original version with random normal initialization for xi.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lambda_ = nn.Parameter(
            torch.normal(mean=0.0, std=1.0,
                         size=(hidden_size, output_size), dtype=torch.cfloat)
        )
        self.xi = nn.Parameter(
            torch.normal(mean=0.0, std=1.0,
                         size=(hidden_size,), dtype=torch.cfloat)
        )
        self.activation = ReciprocalActivation()
    def forward(self, x):
        x_flat = _flatten_input(x)  # => [B]
        B = x_flat.shape[0]

        x_c = torch.complex(x_flat, torch.zeros_like(x_flat))  # => [B]
        x_expanded  = x_c.unsqueeze(1).expand(B, self.hidden_size)
        xi_expanded = self.xi.unsqueeze(0).expand(B, self.hidden_size)

        activated = self.activation(xi_expanded - x_expanded)
        y_complex = torch.matmul(activated, self.lambda_) / self.hidden_size
      
        return y_complex.real

##############################################################################
# 7) CauchyNet_PurelyRealParams
##############################################################################
class CauchyNet_PurelyRealParams(nn.Module):
    """
    Retains a reciprocal-like activation but uses all real parameters.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Real parameters
        self.lambda_ = nn.Parameter(torch.randn(hidden_size, output_size))
        self.xi = nn.Parameter(torch.randn(hidden_size))

        # Reciprocal activation in real domain
        self.activation = ReciprocalActivation()

    def forward(self, x):
        x_flat = _flatten_input(x)
        B = x_flat.shape[0]

        # Expand => [B, hidden_size]
        xi_expanded = self.xi.unsqueeze(0).expand(B, self.hidden_size)
        x_expanded  = x_flat.unsqueeze(-1).expand(B, self.hidden_size)

        out_diff  = xi_expanded - x_expanded
        activated = self.activation(out_diff)  # purely real => shape [B, hidden_size]

        y_real = torch.matmul(activated, self.lambda_) / self.hidden_size
        y_imag = torch.zeros_like(y_real)  # no imaginary part
        return y_real, y_imag


##############################################################################
# 8) CauchyNet_NonHolomorphic
##############################################################################
def complex_relu(z: torch.Tensor) -> torch.Tensor:
    """
    A naive non-holomorphic complex ReLU:
    Re(z) = ReLU(Re(z)), Im(z) = ReLU(Im(z)).
    """
    return torch.complex(
        torch.relu(z.real),
        torch.relu(z.imag)
    )

class CauchyNet_NonHolomorphic(nn.Module):
    """
    Keeps complex parameters but uses a naive 'complex_relu' 
    instead of reciprocal or purely real activations.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Complex parameters
        self.lambda_ = nn.Parameter(
            torch.normal(mean=0.0, std=1.0,
                         size=(hidden_size, output_size), dtype=torch.cfloat)
        )
        self.xi = nn.Parameter(
            torch.normal(mean=0.0, std=1.0,
                         size=(hidden_size,), dtype=torch.cfloat)
        )

    def forward(self, x):
        x_flat = _flatten_input(x)
        B = x_flat.shape[0]

        # Real -> complex
        x_c = torch.complex(x_flat, torch.zeros_like(x_flat))

        # Expand => [B, hidden_size]
        x_expanded  = x_c.unsqueeze(1).expand(B, self.hidden_size)
        xi_expanded = self.xi.unsqueeze(0).expand(B, self.hidden_size)

        diff_z   = xi_expanded - x_expanded
        activated = complex_relu(diff_z)  # shape [B, hidden_size], complex

        y_complex = torch.matmul(activated, self.lambda_) / self.hidden_size
        return y_complex.real, y_complex.imag