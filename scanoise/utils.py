from os.path import join, dirname, basename, splitext

import torch


def replace_dots(fn, to='_'):
    return fn.replace('.', to)


def split_extension(fn):
    """"""
    return join(dirname(fn), splitext(basename(fn))[0])


def to_numpy(tensor):
    """Convert torch.tensor to numpy.ndarray
    """
    if tensor.is_cuda:
        return tensor.data.cpu().numpy()
    else:
        return tensor.data.numpy()


def to_tensor(ndarray):
    """Convert numpy.ndarray to torch.tensor
    """
    return torch.from_numpy(ndarray)


def get_stats(tensor, out_ndarray=True):
    """Get mean / std for tensor.
    
    It flattens the tensor excpet the last dimension
    (assuming it is the feature dimension)

    Args:
        tensor (torch.tensor): input tensor
        out_ndarray (bool): determine wheter convert output to ndarray

    Returns:
        numpy.ndarray or torch.tensor: mean of given vectors
        numpy.ndarray or torch.tensor: std of given vectors
    """
    data = tensor.view(-1, tensor.shape[-1])
    mean, std = data.mean(0), data.std(0)
    if out_ndarray:
        mean, std = to_numpy(mean), to_numpy(std)
    return mean, std
