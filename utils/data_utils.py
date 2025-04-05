import pandas as pd
import os

def load_sst2() -> dict:
    splits = ['train', 'dev', 'test']
    data = {}
    for split in splits:
        file_path = f"data/sst2/{split}.tsv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if split == "test":
            df = pd.read_csv(file_path, sep="\t", header=None, names=["label", "text"])
            df = df[['text']] 
        else:
            df = pd.read_csv(file_path, sep="\t", header=None, names=["label", "text"])
            df = df[['text', 'label']] 
        
        data[split] = df
    return data