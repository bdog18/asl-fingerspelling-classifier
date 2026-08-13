import os
import zipfile
import requests

def download_data(url, save_path):
    """Download dataset from a given URL and save it to the specified path."""
    if not os.path.exists(save_path):
        print(f"Downloading data from {url}...")
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Data already exists.")

def extract_data(zip_path, extract_to):
    """Extract the downloaded zip file to the specified directory."""
    if not os.path.exists(extract_to):
        print(f"Extracting data to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    else:
        print("Data already extracted.")

if __name__ == "__main__":
    # Example usage
    dataset_url = "https://www.kaggle.com/api/v1/datasets/download/debashishsau/aslamerican-sign-language-aplhabet-dataset"
    zip_file_path = "./asl-dataset.zip"
    extract_directory = "./data/raw/ASL_Alphabet_Dataset"

    download_data(dataset_url, zip_file_path)
    extract_data(zip_file_path, extract_directory)