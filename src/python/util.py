import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
import random
import ast

import matplotlib.pyplot as plt
import seaborn as sns

import datetime

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

def check_gpu():
    if torch.cuda.is_available():
        f"GPU is available. Detected {torch.cuda.device_count()} GPU(s)."
        return "cuda"
    else:
        "GPU is not available."
        return "cpu"

def evaluate_model(model, val_loader, criterion, l1_lambda, l1_reg, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # No need to track gradients
        for inputs, targets1, targets2 in val_loader:
            # Forward pass
            outputs1, outputs2 = model(inputs.to(device))

            # Compute loss for each task
            loss1 = criterion(outputs1, targets1.to(device))
            loss2 = criterion(outputs2, targets2.to(device))
            val_loss += loss1.item() + loss2.item() + l1_lambda * l1_reg

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
        print("Loading Encoder Checkpoint")
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

def one_hot_encode(indices:np.ndarray):
    encoded = np.zeros((len(indices),int(indices.max()+1)))
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

def train_val_test_split(x_data, y_data, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    Randomly splits data into training, validation, and test sets using array indexing.
    """
    # Ensure that the sum of the proportions is 1
    assert train_size + val_size + test_size == 1

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

# Function to process the column and return a DataFrame with encoded integers
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

class ModalityEncoder:
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

def set_seed(seed_value):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True  # For consistent results on the GPU
    torch.backends.cudnn.benchmark = False  # Faster convolutions, but might introduce randomness

if __name__ == "__main__":
    with open('secrets.json', 'r') as file:
        secrets = json.load(file)
    file.close()
    
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