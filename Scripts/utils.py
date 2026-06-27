import csv
import torch
import torch.nn as nn

def set_device():
    """
    Configures the default PyTorch device to CUDA if available, or falls back to CPU.
    Sets default data type to float32.
    
    Returns:
        torch.device: The resolved device object.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_device(device)
        print("CUDA is available. Default device set to: cuda")
    else:
        device = torch.device('cpu')
        torch.set_default_device(device)
        print("CUDA is not available. Default device set to: cpu")
    
    torch.set_default_dtype(torch.float32)
    return device



def load_tensors_from_csv(file_path):
    """
    Loads a list of PyTorch tensors from a CSV file.
    Each row in the CSV is expected to have:
    - column 0: tensor index (integer)
    - column 1: tensor shape formatted as "RowsxCols" (e.g., "60x4")
    - column 2+: flattened tensor values
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        list: A list of reconstructed PyTorch tensors.
    """
    tensors = []
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            index = int(row[0])
            shape = tuple(map(int, row[1].split('x')))
            values = list(map(float, row[2:]))
            
            # Reconstruct tensor and reshape to original dimensions
            tensor = torch.tensor(values).reshape(shape)
            tensors.append(tensor)
    return tensors


def get_dense_dimensions(bias_list):
    """
    Extracts the dimensions of the hidden layers from a list of bias tensors.
    This corresponds to the length of each bias tensor, excluding the final output layer.
    
    Args:
        bias_list (list): List of bias tensors.
        
    Returns:
        list: Hidden layers dimensions.
    """
    return [b.shape[0] for b in bias_list[:-1]]


def load_weights_and_biases(model, weights, biases):
    """
    Copies pre-trained weights and biases into the corresponding Linear layers of the model.
    
    Args:
        model (nn.Module): The model instance to load parameters into.
        weights (list): List of weight tensors.
        biases (list): List of bias tensors.
        
    Returns:
        nn.Module: The model with loaded weights and biases.
    """
    with torch.no_grad():
        linear_layer_idx = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Verify weight shape compatibility
                if module.weight.shape == weights[linear_layer_idx].shape:
                    module.weight.copy_(weights[linear_layer_idx])
                else:
                    raise ValueError(
                        f"Shape mismatch for weights at linear layer {linear_layer_idx}: "
                        f"expected {module.weight.shape}, got {weights[linear_layer_idx].shape}"
                    )
                
                # Verify bias shape compatibility
                if module.bias is not None:
                    if module.bias.shape == biases[linear_layer_idx].shape:
                        module.bias.copy_(biases[linear_layer_idx])
                    else:
                        raise ValueError(
                            f"Shape mismatch for bias at linear layer {linear_layer_idx}: "
                            f"expected {module.bias.shape}, got {biases[linear_layer_idx].shape}"
                        )
                linear_layer_idx += 1
    return model
