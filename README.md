This project investigates the feature dynamics of different BERT-based models during training, focusing on modifications to the embedding layer. It includes implementations of standard BERT, a BERT with FFT-based embedding fusion, and a BERT with a 2D convolutional embedding layer, and provides scripts for fine-tuning these models on the SST-2 sentiment analysis dataset and analyzing the learned feature representations.

### Models

The project includes the following BERT-based models for sequence classification:

* **BaseBert:** A standard BERT model for sequence classification, based on the Hugging Face `transformers` library. This serves as a baseline. Available in both from-scratch (`BaseBert.py`) and pre-trained (`BaseBert_pretrained.py`) versions.
* **FFTBert:** A BERT model with a modified embedding layer (`FFTBertEmbeddings`) that uses FFT-based circular convolution to fuse word and position embeddings. Available in both from-scratch (`FFTBert.py`) and pre-trained (`FFTBert_pretrained.py`) versions.
* **ConvBert:** A BERT model that incorporates a 2D convolutional layer in its embedding mechanism. The specific implementation details are in `ConvBert_pretrained.py`, which is imported by `ConvBert.py`. Available in both from-scratch (`ConvBert.py`) and pre-trained (`ConvBert_pretrained.py`) versions.

### Dataset

The project uses the **SST-2 (Stanford Sentiment Treebank v2)** dataset for fine-tuning. The data files (`train.tsv`, `dev.tsv`, `test.tsv`) are expected to be in the `data/sst2` directory. The `utils/data_utils.py` script handles loading and processing this dataset.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd bert_feature_dynamics_final
    ```

2.  **Install dependencies:** The project relies on PyTorch and the Hugging Face `transformers` library, among others. You can install the necessary packages using pip:
    ```bash
    pip install torch transformers numpy tqdm
    ```
    *(Note: Additional dependencies might be required based on the specific analysis scripts used in the `analysis` directory.)*

3.  **Prepare the dataset:** Ensure the SST-2 dataset files (`train.tsv`, `dev.tsv`, `test.tsv`) are placed in the `data/sst2` directory.

### Usage

The main script for fine-tuning the models and extracting features is `train/fine_tuning.py`.

You can run the script with various command-line arguments to configure the training process:

* `--model_type`: Specifies the model to use (`base`, `base_pretrained`, `fft`, `fft_pretrained`, `conv2D`, `conv2D_pretrained`).
* `--num_epochs`: Number of training epochs (default: 25).
* `--batch_size`: Batch size for training and evaluation (default: 32).
* `--learning_rate`: Learning rate for the optimizer (default: 2e-5).
* `--warmup_epochs`: Number of warmup epochs during which some layers can be frozen (default: 0).
* `--freeze_layers`: Number of initial encoder layers to freeze during warmup (0-12) (default: 0).
* `--freeze_embeddings`: Whether to freeze the embedding layer during warmup (1 for True, 0 for False) (default: 0).
* `--early_stop_patience`: Number of epochs to wait for validation loss improvement before early stopping (default: 25).

**Example Commands:**

* Fine-tune the pre-trained BaseBert model for 10 epochs:
    ```bash
    python train/fine_tuning.py --model_type base_pretrained --num_epochs 10
    ```

* Fine-tune the from-scratch FFTBert model with 5 warmup epochs, freezing the first 6 layers and the embeddings:
    ```bash
    python train/fine_tuning.py --model_type fft --num_epochs 25 --warmup_epochs 5 --freeze_layers 6 --freeze_embeddings 1
    ```

The script will save training metrics (loss, accuracy, parameter norms, gradients) and the extracted CLS features for each epoch in a timestamped directory within `/content/drive/MyDrive/bert_feature_outputs` or `./bert_feature_outputs` if the former is not accessible.

### Analysis

The extracted CLS features and training metrics can be used for further analysis of the model's feature dynamics. The `analysis` directory likely contains scripts for visualization (`visualization.py`, `embedding_visualization.py`, `probing_visualization.py`) and probing the learned representations (`probing.py`).

### File Structure

```
.
├── analysis/
│   ├── embedding_visualization.py
│   ├── probing.py
│   ├── probing_visualization.py
│   └── visualization.py
├── data/
│   ├── agnews/
│   │   ├── test.csv
│   │   └── train.csv
│   └── sst2/
│       ├── dev.tsv
│       ├── test.tsv
│       └── train.tsv
├── models/
│   ├── BaseBert.py
│   ├── BaseBert_pretrained.py
│   ├── BaseBert copy.py  # (Possibly a backup or alternative version)
│   ├── ConvBert.py
│   ├── ConvBert_pretrained.py
│   ├── FFTBert.py
│   └── FFTBert_pretrained.py
├── train/
│   └── fine_tuning.py
└── utils/
    ├── data_utils.py
    └── train_utils.py
```