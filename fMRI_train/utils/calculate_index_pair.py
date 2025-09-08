import torch
import numpy as np

def generate_unique_pairs(n, e_1, device='cuda' if torch.cuda.is_available() else 'cpu', seed=None):
    """
    Generate unique unordered pairs across multiple epochs using PyTorch.

    :param n: Number of indices (0 to n-1).
    :param e: Number of epochs.
    :param device: Device to perform computations on ('cuda' or 'cpu').
    :param seed: Random seed for reproducibility (optional).
    :return: Tensor of shape (e, n, 2), containing the pairs for each epoch.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    e=e_1+1
    # Validate the number of epochs
    max_epochs = (n - 1) // 2
    if e > max_epochs:
        # Generate pairs for the maximum number of epochs first
        max_epochs = (n - 1) // 2
        max_epoch_pairs = generate_unique_pairs(n, max_epochs, device=device, seed=seed).to(device)



        # For the remaining epochs, generate random pairs
        remaining_epochs = e - max_epochs
        random_pairs = torch.empty((remaining_epochs, n, 2), dtype=torch.long, device=device)

        for epoch in range(remaining_epochs):
            random_pairs[epoch, :, 0] = torch.arange(n, device=device)
            random_pairs[epoch, :, 1] = torch.randint(0, n, (n,), device=device)
            mask = random_pairs[epoch, :, 1] == torch.arange(n, device=device)
            random_pairs[epoch, mask, 1] = (random_pairs[epoch, mask, 1] + 1) % n

        # Combine the max_epoch_pairs and random_pairs
        result = torch.cat([max_epoch_pairs, random_pairs], dim=0)

        return result.cpu()


    used_pairs = torch.zeros((n, n), dtype=torch.bool, device=device)

    result = torch.empty((e, n, 2), dtype=torch.long, device=device)

    for epoch in range(e):
        available_js = (~used_pairs) & (~torch.eye(n, dtype=torch.bool, device=device))
        available_js = (~used_pairs) & (~torch.eye(n, dtype=torch.bool, device=device))

        rand_scores = torch.rand((n, n), device=device)
        rand_scores[~available_js] = -1  # Assign a low score to unavailable pairs

        _, selected_js = torch.max(rand_scores, dim=1)

        result[epoch, :, 0] = torch.arange(n, device=device)  # First elements: 0 to n-1
        result[epoch, :, 1] = selected_js  # Second elements: selected_js

        used_pairs[torch.arange(n, device=device), selected_js] = True
        used_pairs[selected_js, torch.arange(n, device=device)] = True

    return result.cpu()

# Example Usage
if __name__ == "__main__":
    import time

    # Define parameters
    n = 80  # Number of indices (0 to 9999)
    e = 50     # Number of epochs
    seed = 42  # For reproducibility

    # Generate pairs
    start_time = time.time()
    pairs = generate_unique_pairs(n, e, seed=seed)
    end_time = time.time()

    epoch_to_inspect = 24 
    epoch_pairs = pairs[epoch_to_inspect]

    print(f"Sample Pairs from Epoch {epoch_to_inspect + 1}:")
    print(epoch_pairs[:10]) 

    