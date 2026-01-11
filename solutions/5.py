import torch

def scalar_multiply(matrix, scalar) -> torch.Tensor:
    """
    Multiply each element of a 2D matrix by a scalar using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of the same shape.
    """
    # Convert input to tensor
    m_t = torch.as_tensor(matrix, dtype=torch.float)
    return m_t * scalar


"""
Note: Uses
1. Broadcasting : Scalar multiplication is a basic form of broadcasting where the single scalar value is automatically "expanded" to match the shape of the matrix.
2. Type Promotion : If you multiply an integer tensor by a float scalar, PyTorch typically promotes the result to a floating-point tensor to maintain precisio
"""