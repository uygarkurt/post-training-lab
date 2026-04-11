"""
Data preparation scripts for various reasoning datasets.

Each dataset has its own module with:
- DATASET_NAME: The official dataset identifier (e.g., "openai/gsm8k")
- prepare_data(): Function that returns (train_dataset, val_dataset)
"""

# Import all dataset modules
from . import gsm8k
from . import openthoughts_114k
from . import openmathinstruct
from . import magpie_reasoning

# Collect all available modules
_MODULES = [gsm8k, openthoughts_114k, openmathinstruct, magpie_reasoning]

def get_prep_module(dataset_name: str):
    """
    Get the preparation module for a given dataset.
    
    Args:
        dataset_name: The name of the dataset (e.g., "openai/gsm8k")
        
    Returns:
        The module with prepare_data() function
        
    Raises:
        ValueError: If dataset is not supported
    """
    if dataset_name == "openai/gsm8k":
        return gsm8k
    elif dataset_name == "open-thoughts/OpenThoughts-114k":
        return openthoughts_114k
    elif dataset_name == "nvidia/OpenMathInstruct-2":
        return openmathinstruct
    elif dataset_name == "Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B":
        return magpie_reasoning
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported.\n"
            "To add support, create a new module in data_prep/"
        )
