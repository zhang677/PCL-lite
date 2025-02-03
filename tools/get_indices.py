from functools import reduce
import torch

def get_full_indices(shape):
    total_elements = reduce(lambda x, y: x * y, shape)
    indices = []
    for i in range(total_elements):
        idx = []
        for j in range(len(shape)):
            idx.append(i % shape[j])
            i //= shape[j]
        indices.append(tuple(idx[::-1]))
    return indices

# print(get_full_indices([4,]))
# print(get_full_indices([3, 2]))
# print(get_full_indices([2, 3, 4]))

def generate_multi_hot(size, min_ones, max_ones, e):
    # Create a tensor of zeros with shape (*size, e)
    if isinstance(size, int):
        size = (size,)
    tensor = torch.zeros(size + (e,))
    
    # Calculate total number of vectors needed
    total_vectors = torch.prod(torch.tensor(size)).item()
    
    # Reshape tensor for easier indexing
    reshaped_tensor = tensor.view(-1, e)
    
    # Generate multi-hot vectors for each position
    for i in range(total_vectors):
        # Generate random number of ones between min_ones and max_ones
        ones = torch.randint(min_ones, max_ones + 1, (1,)).item()
        # Set random positions to 1
        reshaped_tensor[i, torch.randperm(e)[:ones]] = 1
    
    # Reshape back to original dimensions
    return tensor

def generate_binary_tensor(size, p=0.5):
    return torch.bernoulli(torch.ones(size) * p)

def multihot_to_onehot_list(multihot_tensor):
    num_classes = multihot_tensor.size(-1)
    return [
        torch.where(multihot_tensor[..., i].flatten())
        for i in range(num_classes)
    ]

# multih_tensor = generate_multi_hot((2, 3), 0, 2, 4)
# print(multih_tensor)
# onehot_list = multihot_to_onehot_list(multih_tensor)
# print(onehot_list)

def topk_to_mutihot(affinity, k):
    _, indices = torch.topk(affinity, k, dim=0)
    multihot = torch.zeros_like(affinity, dtype=torch.float32)
    multihot.scatter_(0, indices, 1.0)
    return multihot

affinity = torch.randn(5, 3)
k = 2
multihot = topk_to_mutihot(affinity, k)
print(affinity)
print(multihot)