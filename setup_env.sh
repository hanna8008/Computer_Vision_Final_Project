#!/bin/bash


# --- Setup Envrionment Script ---
ENV_NAME="fitme-env"
PYTHON_VERSION=3.9

echo "Creating Conda envrionment: $ENV_NAME..."



# --- Initialize Conda
CONDA_BASE="$HOME/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"



# --- Create Envrionment (if not exists) ---
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "New Conda environment '$ENV_NAME' is being created with Python $PYTHON_VERSION."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi



# --- Install Requirements ---
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete for environment '$ENV_NAME'."


# --- Activate Envrionment ---
conda activate $ENV_NAME
echo "Envrionment $ENV_NAME activated."