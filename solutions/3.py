import torch

def reshape_matrix(a, new_shape) -> torch.Tensor:
    """
    Reshape a 2D matrix `a` to shape `new_shape` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a tensor of shape `new_shape`, or an empty tensor on mismatch.
    """
    try:
        a_t = torch.as_tensor(a, dtype=torch.float)
        return a_t.reshape(new_shape)
    except RuntimeError:
        return torch.tensor([])
    



"""
Note:
This can also be achieved using basic python :
1. check for compatibility
2. flatten the list using list comprehension 
3. create the reshape matrix by slicing the flat list in a loop
"""