import torch  # Must be imported before pandas on Windows to avoid DLL loading conflicts
import os
import pandas as pd
from Scripts import models
from Scripts import utils


# Pre-trained Model Paths
MODEL_DIR = os.path.join("Models", "NN_Dual_frequency")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "NN_Dual_frequency_Weights.csv")
BIASES_PATH = os.path.join(MODEL_DIR, "NN_Dual_frequency_Biases.csv")

# Input/Output Data Paths
# This path can be configured to point to your new input data file (.csv or .xlsx)
INPUT_DATA_PATH = os.path.join("Inputs", "input_data_2h.csv")
OUTPUT_CSV_PATH = os.path.join("Outputs", "predictions_2h.csv")




def load_input_data(file_path, device):
    """
    Loads raw input data from a CSV or Excel file.
    According to specifications, the columns are:
    - Column 0: Real part (1H)
    - Column 1: Imaginary part (1H)
    - Column 2: Real part (2H)
    - Column 3: Imaginary part (2H)
    
    Args:
        file_path (str): Path to the input CSV or Excel file.
        device (torch.device): Device to load the data tensor onto.
        
    Returns:
        torch.Tensor: Preprocessed input tensor of shape (N, 4) on the correct device.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Input data file not found at: '{file_path}'. "
            f"Please update the INPUT_DATA_PATH variable with your data file's path."
        )

    print(f"Loading input data from: {file_path}")
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
        
    # Extract the first four columns (Real 1H, Imag 1H, Real 2H, Imag 2H)
    raw_values = df.iloc[:, :4].values
    
    # Convert to PyTorch tensor and move to device
    input_tensor = torch.tensor(raw_values, dtype=torch.float32).to(device)
    return input_tensor


def main():
    # 1. Setup PyTorch device and default settings
    device = utils.set_device()
    
    # Verify pre-trained parameter files exist
    if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(BIASES_PATH):
        raise FileNotFoundError(
            f"Pre-trained weights or biases not found in model folder: {MODEL_DIR}. "
            f"Ensure the path is relative to the directory containing this script."
        )
        
    # 2. Load pre-trained weights and biases
    print("Loading pre-trained model weights and biases...")
    weights = utils.load_tensors_from_csv(WEIGHTS_PATH)
    biases = utils.load_tensors_from_csv(BIASES_PATH)
    
    # Extract the dense layer dimensions based on bias shapes
    dense_dimensions = utils.get_dense_dimensions(biases)
    print(f"Loaded architecture layer dimensions: {dense_dimensions}")
    
    # 3. Initialize model and copy pre-trained weights/biases
    # Dropout rate is set to 0.01 for inference as configured in the original script
    model = models.DualFrequencyModel(layer_dims=dense_dimensions, dropout=0.01)
    model = utils.load_weights_and_biases(model, weights, biases)
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    # 4. Load the input dataset
    try:
        input_tensor = load_input_data(INPUT_DATA_PATH, device)
    except FileNotFoundError as e:
        print(f"\n[Warning] {e}")
        print("Please place the input file or edit INPUT_DATA_PATH at the top of this script.")
        return
        
    print(f"Input data shape: {list(input_tensor.shape)}")
    
    # 5. Run inference
    print("Running model inference...")
    with torch.no_grad():
        predictions = model(input_tensor)
        # Round the predictions to integer values as in original script
        predictions = torch.round(predictions)
        
    # Move predictions to CPU and convert to NumPy array
    predictions_np = predictions.cpu().numpy()
    
    # 6. Post-process to compute absolute X2 position
    # The raw model output is [X1, Y1, X2-X1, Y2]
    # To get X2, we add X1 (column 0) to X2-X1 (column 2)
    print("Post-processing predictions: Calculating absolute X2 by adding X1...")
    predictions_np[:, 2] = predictions_np[:, 2] + predictions_np[:, 0]
    
    # 7. Save outputs to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    predictions_df = pd.DataFrame(predictions_np, columns=["X1", "Y1", "X2", "Y2"])
    predictions_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Successfully saved inference predictions to: {OUTPUT_CSV_PATH}")
    print(f"Sample predictions (first 5 rows):\n{predictions_df.head()}")



if __name__ == '__main__':
    main()
