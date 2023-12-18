"""
    Frequently used functions
    If viewing with VScode, alt + left click function collapse arrow to collapse all
    Function groups labelled for ease of understanding
"""

import pandas as pd
import numpy as np

from sqlalchemy import create_engine
import numpy as np
from itertools import product
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
import random
import ast

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

import datetime
# Output and export
def export_to_onnx(onnx_base_path, onnx_name, model:nn.Module, checkpoint, input_size):
    # write brackets model
    model.load_state_dict(
        checkpoint['model_state_dict']
    )
    print(model)
    x = torch.randn(1, 1, input_size, requires_grad=False)
    torch_out = model(x)
    torch.onnx.export(model,                   # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    f'{onnx_base_path}{onnx_name}.onnx',   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
    print(f'File saved to: {onnx_base_path}{onnx_name}')

# Visualisation and exploration
def plot_class_histograms(Y):
    # Plotting the histogram for the first column of Y_train
    plt.figure(figsize=(10, 6))
    sns.histplot(Y, kde=False, color="skyblue", bins=10)
    plt.title('Histogram of Y_train First Column')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

def add_dirichlet_noise(inputs, alpha, noise_ratio):
    # Generate Dirichlet noise
    dirichlet_noise = np.random.dirichlet(alpha, size=inputs.shape[0])

    # Scale and shift noise to match input range
    dirichlet_noise = 2 * dirichlet_noise - 1

    # Convert to tensor
    dirichlet_noise = torch.tensor(dirichlet_noise, dtype=inputs.dtype)

    # Mix noise with inputs
    noisy_inputs = (1 - noise_ratio) * inputs + noise_ratio * dirichlet_noise
    return noisy_inputs

def find_latest_file(directory):
    """ Find the latest file in the given directory. """
    # Ensure the directory exists
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    # List all files in the directory
    files = [os.path.join(directory, file) for file in os.listdir(directory)]
    
    # Filter out directories, only keep files
    files = [f for f in files if os.path.isfile(f)]

    # Ensure there are files in the directory
    if not files:
        return None

    # Find the file with the latest modification time
    latest_file = max(files, key=os.path.getmtime)

    return latest_file

def capitalize_words(input_string, delimiter):
    # Split the string by the specified delimiter
    words = input_string.split(delimiter)
    
    # Capitalize the first letter of each word
    capitalized_words = [word.capitalize() for word in words]
    
    # Join the words back into a string
    new_string = ' '.join(capitalized_words)
    
    return new_string
        
def time_metric_boxplot(x_column: str, y_column_1: str, y_column_2: str, hue: str, dataframe: pd.DataFrame, show = False):
    plt.figure(figsize=(14, 6))

    # Monthly trends for first_pgsi
    plt.subplot(1, 2, 1)
    sns.boxplot(x=x_column, y=y_column_1, hue=hue, data=dataframe)
    plt.title(f'{x_column.capitalize()}ly Trends in {capitalize_words(y_column_1, "_")}')
    plt.xlabel(f'{x_column.capitalize()}')
    plt.ylabel(f'{capitalize_words(y_column_1, "_")}')
    plt.legend(title=f'{capitalize_words(hue, "_")}', loc='upper right')

    # Monthly trends for first_core10
    plt.subplot(1, 2, 2)
    sns.boxplot(x=x_column, y=y_column_2, hue=hue, data=dataframe)
    plt.title(f'{x_column.capitalize()}ly Trends in {capitalize_words(y_column_2, "_")}')
    plt.xlabel(f'{x_column.capitalize()}')
    plt.ylabel(f'{capitalize_words(y_column_2, "_")}')
    plt.legend(title=f'{capitalize_words(hue, "_")}', loc='upper right')

    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(f'./visualisation-plots/{x_column}_{y_column_1}_{y_column_2}.png')
    
def time_scatterplot(x_column: str, y_column_1: str, y_column_2: str, dataframe: pd.DataFrame, show = False):
    plt.figure(figsize=(14, 6))

    # Monthly trends for first_pgsi
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=x_column, y=y_column_1, data=dataframe, s = 3.5)
    plt.title(f'Relationship between the {capitalize_words(x_column, "_")} and {capitalize_words(y_column_1, "_")}')
    plt.xlabel(f'{capitalize_words(x_column, "_")}')
    plt.ylabel(f'{capitalize_words(y_column_1, "_")}')

    # Monthly trends for first_core10
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=x_column, y=y_column_2, data=dataframe, s = 3.5)
    plt.title(f'Relationship between the {capitalize_words(x_column, "_")} and {capitalize_words(y_column_2, "_")}')
    plt.xlabel(f'{capitalize_words(x_column, "_")}')
    plt.ylabel(f'{capitalize_words(y_column_2, "_")}')

    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(f'./visualisation-plots/{x_column}_{y_column_1}_{y_column_2}.png')

# System checks
def check_gpu():
    if torch.cuda.is_available():
        f"GPU is available. Detected {torch.cuda.device_count()} GPU(s)."
        return "cuda"
    else:
        "GPU is not available."
        return "cpu"

def set_seed(seed_value):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True  # For consistent results on the GPU
    torch.backends.cudnn.benchmark = False  # Faster convolutions, but might introduce randomness

# Evaluation functions
def evaluate_model(model, val_loader, criterion, l1_lambda, l1_reg, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # No need to track gradients
        for inputs, targets1, targets2 in val_loader:
            # Forward pass
            outputs1, outputs2 = model(inputs.to(device))

            # Compute loss for each task
            loss1 = criterion[0](outputs1, targets1.to(device))
            loss2 = criterion[1](outputs2, targets2.to(device))
            val_loss += loss1.item() + loss2.item() # + l1_lambda * l1_reg
            
            # dual focal loss
            # loss = criterion([outputs1, outputs2], [targets1, targets2])
            # val_loss = loss + l1_lambda * l1_reg

    return val_loss / len(val_loader)

def evaluate_modality_model(model, val_loader, criterion, l1_lambda, l1_reg, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # No need to track gradients
        for i, data_batch in enumerate(val_loader):
            inputs, targets = data_batch
            
            outputs = model(inputs.to(device))
            l1_reg = sum(param.abs().sum() for param in model.parameters())

            # Compute loss for each task
            loss = criterion(outputs, targets.to(device))
            val_loss += loss.item() # + l1_lambda * l1_reg
            

    return val_loss / len(val_loader)

def save_checkpoint(current_epoch, model, optimiser, loss, checkpoint_file='./'):
    torch.save({
        'ganme' : current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'loss (BCE w/ Logits)': loss,
    }, checkpoint_file)
    print(f'Saving model at epoch {current_epoch}')

def load_checkpoint(directory):
    if os.path.isfile(directory):
        print("Loading Model Checkpoint")
        checkpoint = torch.load(directory)
        return checkpoint
    else:
        pass

def model_summary(model, input_size):
    print("----------------------------------------------------------------")
    print(f"Input Shape:               {str(input_size).ljust(25)}")
    print("----------------------------------------------------------------")
    print("Layer (type)               Output Shape         Param #")
    print("================================================================")
    
    total_params = 0

    def register_hook(module):
        def hook(module, input, output):
            nonlocal total_params
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params

            # Remove torch.Size
            if isinstance(output, tuple):
                output_shape = [str(list(o.shape)) if torch.is_tensor(o) else str(type(o)) for o in output]
                # Pick first size if there are multiple identical sizes in the tuple
                output_shape = output_shape[0]
            else:
                output_shape = str(list(output.shape))

            if len(list(module.named_children())) == 0:  # Only print leaf nodes
                print(f"{module.__class__.__name__.ljust(25)}  {output_shape.ljust(25)} {f'{num_params:,}'}")

        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    model.apply(register_hook)

    print("----------------------------------------------------------------")
    DEVICE = next(model.parameters()).device
    output = model(torch.randn(1, *input_size).to(DEVICE))

    for h in hooks:
        h.remove()

    output_shape = str(list(output.shape)) if torch.is_tensor(output) else str(type(output))
    print("----------------------------------------------------------------")
    print(f"Total params: {total_params:,}")
    print(f"Output Shape: {output_shape.ljust(25)}")
    print(f'Model on: {next(model.parameters()).device}')
    print("----------------------------------------------------------------")

# Database handling requests
def get_data(username, password, node_ip, port, db_name, query_path) -> pd.DataFrame:
    postgres_url = f'postgresql://{username}:{password}@{node_ip}:{port}/{db_name}'
    engine = create_engine(postgres_url)
    try:
        with open(query_path, 'r') as file:
            query = file.read()
        # Using pandas.read_sql to execute the query and load data into a DataFrame
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

    finally:
        file.close()
        engine.dispose()

def write_to_sql(username, password, node_ip, port, db_name, table_name, excel_file: None, dataframe: None):
    # Define the PostgreSQL URL
    postgres_url = f'postgresql://{username}:{password}@{node_ip}:{port}/{db_name}'

    # Create an engine
    engine = create_engine(postgres_url)

    df = None

    if excel_file is not None and dataframe is None:
        df = pd.read_excel(excel_file)
    elif dataframe is not None and excel_file is None:
        df = dataframe
    else:
        raise ValueError("Please provide either an Excel file or a DataFrame, not both or neither.")

    try:
        # Write records stored in a DataFrame to a SQL database
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Data pushed successfully to table '{table_name}' in database '{db_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Data processing
def one_hot_encode(indices:np.ndarray):
    encoded = np.zeros((len(indices),int(indices.max()+1)))
    print(encoded, encoded.shape)
    encoded[np.arange(len(indices)), indices.astype(int)] = np.float32(1)
    return encoded, encoded.shape[1]

def selective_median_fill(df: pd.DataFrame, columns : list)->pd.DataFrame:
    df[columns] = df[columns].fillna(df[columns].median())
    return df

def standardize_data(X):
    """
    Standardizes each column of the numerical data and returns the means and standard deviations.
    """
    means = np.mean(X, axis=0)
    # print(means)
    stds = np.std(X, axis=0)
    # print(stds)
    # Avoid division by zero in case of a constant column
    stds[stds == 0] = 1

    X_standardized = (X - means) / stds
    return X_standardized, means, stds

def stratified_train_val_test_split(x_data, y_data, train_size=0.7, val_size=0.2, test_size=0.1, stratify_on=0):
    """
    Splits data into training, validation, and test sets using stratified sampling based on one set of class indices.
    """
    assert abs(train_size + val_size + test_size - 1) < 1e-6

    # Choose which set of class indices to use for stratification
    y_stratify = y_data

    # Define the stratified split for train and remaining data
    stratified_split_train = StratifiedShuffleSplit(n_splits=1, test_size=1-train_size, random_state=42)
    train_indices, remaining_indices = next(stratified_split_train.split(x_data, y_stratify))

    X_train, Y_train = x_data[train_indices], y_data[train_indices]
    X_remaining, Y_remaining = x_data[remaining_indices], y_data[remaining_indices]

    # Further split remaining data into validation and test sets
    remaining_size = val_size / (val_size + test_size)
    stratified_split_val_test = StratifiedShuffleSplit(n_splits=1, test_size=remaining_size, random_state=42)
    val_indices, test_indices = next(stratified_split_val_test.split(X_remaining, Y_remaining[:, stratify_on]))

    X_val, Y_val = X_remaining[val_indices], Y_remaining[val_indices]
    X_test, Y_test = X_remaining[test_indices], Y_remaining[test_indices]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def train_val_test_split(x_data, y_data, train_size=0.7, val_size=0.2, test_size=0.1):
    """
    Randomly splits data into training, validation, and test sets using array indexing.
    """
    # Ensure that the sum of the proportions is 1
    assert abs(train_size + val_size + test_size - 1) < 1e-6

    # Shuffle the data
    indices = np.arange(x_data.shape[0])
    np.random.shuffle(indices)
    x_data, y_data = x_data[indices], y_data[indices]

    # Number of samples
    num_samples = x_data.shape[0]

    # Calculate indices for splits
    train_end = int(train_size * num_samples)
    val_end = train_end + int(val_size * num_samples)

    # Split data into training, validation, and test sets
    X_train, Y_train = x_data[:train_end], y_data[:train_end]
    X_val, Y_val = x_data[train_end:val_end], y_data[train_end:val_end]
    X_test, Y_test = x_data[val_end:], y_data[val_end:]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def get_class_frequencies_multi_target(data_loader, num_classes1, num_classes2):
    # Initialize counters for each set of target columns
    class_counts1 = torch.zeros(num_classes1, dtype=torch.float32)
    class_counts2 = torch.zeros(num_classes2, dtype=torch.float32)

    # Iterate over the data loader
    for i, data_batch in enumerate(data_loader):  # Assuming each batch returns (data, (targets1, targets2))
        inputs, targets1, targets2 = data_batch
        for label in targets1:
            class_counts1[label] += 1
        for label in targets2:
            class_counts2[label] += 1

    return class_counts1, class_counts2

def calculate_alpha(frequencies: torch.Tensor):
    # Calculate inverse frequency
    weights = frequencies.sum() / (frequencies * len(frequencies))
    # inverse_freq = 1.0 / frequencies
    # Normalize so that alphas sum to the number of classes
    # return inverse_freq / inverse_freq.sum() * len(frequencies)
    return weights

def generate_label_combinations(labels1, labels2):
    """
    Generate all possible combinations of two sets of labels.
    """
    return list(product(labels1, labels2))

def find_min_samples_per_combination(Y, label_combinations):
    """
    Find the minimum number of samples for any label combination in Y.
    """
    label_counts = Counter(tuple(row) for row in Y)
    min_samples = float('inf')
    for comb in label_combinations:
        min_samples = min(min_samples, label_counts.get(comb, 0))
    return min_samples

def random_undersampling_multilabel(X, Y, label_combinations, min_samples):
    """
    Randomly undersample the dataset to even out all label combinations.
    """
    indices_to_select = []
    for label_comb in label_combinations:
        label_indices = [i for i, label in enumerate(Y) if tuple(label) == label_comb]
        selected_indices = np.random.choice(label_indices, min(min_samples, len(label_indices)), replace=False)
        indices_to_select.extend(selected_indices)

    return X[indices_to_select], Y[indices_to_select]

def undersampling_train_val_test_split(x_data, y_data, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    Randomly splits data into training, validation, and test sets using array indexing,
    with an option for random undersampling in a multi-label context.
    """
    # Ensure that the sum of the proportions is 1
    assert train_size + val_size + test_size == 1

    # Generate all label combinations
    label_combinations = generate_label_combinations([0, 1, 2, 3], [0, 1, 2, 3, 4])

    # Find the minimum number of samples for any label combination
    min_samples = find_min_samples_per_combination(y_data, label_combinations)

    # Apply random undersampling
    x_data, y_data = random_undersampling_multilabel(x_data, y_data, label_combinations, min_samples)

    # Shuffle the data
    indices = np.arange(x_data.shape[0])
    np.random.shuffle(indices)
    x_data, y_data = x_data[indices], y_data[indices]

    # Split data into training, validation, and test sets
    X_train, X_test_val, Y_train, Y_test_val = train_test_split(x_data, y_data, train_size=train_size, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test_val, Y_test_val, train_size=val_size/(val_size + test_size), random_state=42)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

# Classification encoders
class InitialModalityEncoder:
    def __init__(self, column_name):
        self.column_name = column_name
        self.element_to_int = None
        self.int_to_element = None

    def fit(self, df):
        """Fit the encoder to the DataFrame."""
        first_elements = df[self.column_name].apply(self.extract_first_element)

        # Extract unique elements
        unique_elements = sorted(set(first_elements))

        # Assign IDs based on the sorted order of unique elements
        self.element_to_int = {element: idx for idx, element in enumerate(unique_elements)}
        self.int_to_element = {idx: element for element, idx in self.element_to_int.items()}

    def transform(self, df)->pd.DataFrame:
        """Transform the DataFrame using the fitted encoder."""
        if self.element_to_int is None:
            raise Exception("FirstElementEncoder not fitted. Please call 'fit' with the DataFrame first.")
        
        df['first_element'] = df[self.column_name].apply(self.extract_first_element)
        df['first_element_int'] = df['first_element'].apply(lambda x: self.element_to_int[x])
        return df

    def get_lookup_table(self)->pd.DataFrame:
        """Get the lookup table for the first elements."""
        if self.int_to_element is None:
            raise Exception("FirstElementEncoder not fitted. Lookup table not available.")
        
        # Creating a list of dictionaries for each element and its corresponding ID
        lookup_data = [{'first_element': element, 'element_id': idx} 
                       for element, idx in self.element_to_int.items()]

        # Converting to DataFrame
        return pd.DataFrame(lookup_data)
    
    @staticmethod
    def extract_first_element(modality_str):
        """Extract and return the first element of the modality list."""
        modality_list = ast.literal_eval(modality_str)
        # Return the first phrase (element) from the list
        return modality_list[0] if modality_list else None

class ComboModalityEncoder:
    def __init__(self, column_name):
        self.column_name = column_name
        self.set_to_int = None
        self.int_to_set = None

    def fit(self, df):
        """Fit the encoder to the DataFrame."""
        cleaned_lists = df[self.column_name].apply(self.clean_modality_type)
        sets = cleaned_lists.apply(frozenset)

        # Extract unique sets and sort them based on a string representation
        unique_sets = sorted(set(sets), key=lambda s: ','.join(sorted(s)))

        # Assign IDs based on the sorted order of unique sets
        self.set_to_int = {mod_set: idx for idx, mod_set in enumerate(unique_sets)}
        self.int_to_set = {idx: mod_set for mod_set, idx in self.set_to_int.items()}

    def transform(self, df)->pd.DataFrame:
        """Transform the DataFrame using the fitted encoder."""
        if self.set_to_int is None:
            raise Exception("ModalityEncoder not fitted. Please call 'fit' with the DataFrame first.")
        
        df['modality_type_cleaned'] = df[self.column_name].apply(self.clean_modality_type)
        df['modality_type_set'] = df['modality_type_cleaned'].apply(frozenset)
        df['modality_type_int'] = df['modality_type_set'].apply(lambda x: self.set_to_int[x])
        return df

    def get_lookup_table(self)->pd.DataFrame:
        """Get the lookup table for the modality combinations."""
        if self.int_to_set is None:
            raise Exception("ModalityEncoder not fitted. Lookup table not available.")
        
        # Creating a list of dictionaries for each modality set and its corresponding ID
        lookup_data = [{'modality_combination': ', '.join(sorted(list(mod_set))), 'modality_id': idx} 
                       for mod_set, idx in self.set_to_int.items()]

        # Converting to DataFrame
        return pd.DataFrame(lookup_data)

    @staticmethod
    def clean_modality_type(modality_str):
        """Clean and standardize the modality type."""
        return sorted(set(ast.literal_eval(modality_str)))

# Deprivated and replaced by class encoding classes (above)
def process_modality_type_column(modality_column):
    # Clean and standardize the modality type
    cleaned_lists = modality_column.apply(lambda x: sorted(set(ast.literal_eval(x))))

    # Extract all unique elements
    unique_elements = set()
    cleaned_lists.apply(unique_elements.update)

    # Create a mapping from elements to integers
    element_to_int = {element: idx for idx, element in enumerate(unique_elements)}

    # Replace elements with integers
    encoded_lists = cleaned_lists.apply(lambda lst: [element_to_int[element] for element in lst])

    # Return as a DataFrame
    return pd.DataFrame({'modality_type_encoded': encoded_lists})

# Custom focal loss class - eventually replaced with torch weighted cross entropy loss
class FocalLoss(nn.Module):
    def __init__(self, alpha1=None, alpha2=None, gamma=2.0, num_classes1=4, num_classes2=5, device='cpu'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.device = device

        if alpha1 is None:
            self.alpha1 = torch.ones(num_classes1).to(device)
        else:
            self.alpha1 = alpha1.to(device)

        if alpha2 is None:
            self.alpha2 = torch.ones(num_classes2).to(device)
        else:
            self.alpha2 = alpha2.to(device)

    def forward(self, inputs, targets):
        # Calculate loss for the first output
        loss1 = self.calculate_loss(inputs[0], targets[0], self.alpha1, self.num_classes1)

        # Calculate loss for the second output
        loss2 = self.calculate_loss(inputs[1], targets[1], self.alpha2, self.num_classes2)

        # Average the losses
        return (loss1 + loss2) / 2

    def calculate_loss(self, input, target, alpha, num_classes):
        # Ensure everything is on the same device
        input = input.to(self.device)
        target = target.to(self.device)

        # Compute the softmax
        softmax = F.softmax(input, dim=1)
        
        # Create one hot encoding of targets
        target_one_hot = F.one_hot(target, num_classes=num_classes).float().to(self.device)

        # Calculate Focal Loss
        pt = torch.sum(target_one_hot * softmax, dim=1)
        alpha = alpha.expand_as(softmax)

        # Select the alphas for the target classes
        alpha = torch.gather(alpha, 1, target.unsqueeze(1))
        alpha = alpha.squeeze(1)

        focal_loss = -alpha * torch.pow(1 - pt, self.gamma) * torch.log(pt + 1e-8)
        return focal_loss.mean()

if __name__ == "__main__":
    with open('secrets.json', 'r') as file:
        secrets = json.load(file)
    file.close()
    # Testing SQL server connection
    # Accessing the configuration values
    connection_params = {
        "username": secrets['sqlConfig']['username'],
        "password": secrets['sqlConfig']['password'],
        "port": secrets['sqlConfig']['port'],
        "node_ip": secrets['sqlConfig']['ip'],
        "db_name": secrets['sqlConfig']['db_name']
    }
    table_name = secrets['sqlConfig']['modality_table']
    excel_file = secrets['excelPath']
    
    test = get_data(**connection_params, query_path=secrets["sqlQueryFilePaths"]["get_first_pgci_core10_referral"])
    print(test, test.shape)