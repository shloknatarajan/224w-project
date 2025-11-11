# CS224W Project: Link Prediction on OGBL-DDI

This project implements and compares three Graph Neural Network (GNN) architectures for link prediction on the OGBL-DDI (Drug-Drug Interaction) dataset.

## Overview

The project performs link prediction using three different GNN models:
- **GCN** (Graph Convolutional Network)
- **GraphSAGE** (Graph Sample and Aggregate)
- **GraphTransformer** (Transformer-based GNN)

All models are evaluated using the Hits@20 metric on the OGBL-DDI dataset.

## Key Features

- **No Data Leakage**: The graph is constructed using only training edges to prevent information leakage
- **Learnable Node Embeddings**: Uses trainable embeddings for stable feature learning
- **Early Stopping**: Validation-based early stopping to prevent overfitting
- **Comprehensive Logging**: Results are logged to timestamped files in the `logs/` directory
- **GPU Support**: Automatically uses CUDA if available, falls back to CPU otherwise
- **Batch Normalization & Dropout**: Regularization techniques for better generalization

## Dataset

The project uses the **ogbl-ddi** (Drug-Drug Interaction) dataset from the Open Graph Benchmark (OGB).

### Overview

The ogbl-ddi dataset is a homogeneous, unweighted, and undirected graph representing drug-drug interactions. Each node corresponds to an FDA-approved or experimental drug, and an edge between two nodes signifies a significant interaction between the respective drugs. A drug-drug interaction indicates that the combined effect of taking the two drugs together differs substantially from their independent effects, which is crucial information for pharmaceutical research and patient safety.

### Dataset Statistics

- **Number of Nodes (Drugs)**: 4,267
- **Number of Edges (Interactions)**: 1,334,889
- **Average Node Degree**: ~500.5
- **Graph Density**: ~14.67%

### Prediction Task

The primary objective is to predict potential drug-drug interactions based on known interactions. This is a link prediction task where the model learns to identify which drug pairs are likely to interact. The evaluation metric **Hits@20** assesses the model's ability to rank true drug interactions higher than non-interacting drug pairs. Specifically, each true drug interaction is ranked among approximately 100,000 randomly-sampled negative interactions, and the metric counts the ratio of positive edges ranked at 20th place or above.

### Data Splitting Strategy

The dataset employs a **protein-target split** strategy. This means that drug edges are divided based on the proteins those drugs target. Consequently, the test set comprises drugs that predominantly bind to different proteins than those in the training and validation sets. This approach evaluates the model's ability to generalize to drugs with distinct biological mechanisms of action, which is critical for real-world applications where new drugs with novel targets need to be evaluated.

### Importance

Predicting drug-drug interactions is essential for:
- **Drug Discovery**: Identifying potential interactions early in the drug development process
- **Patient Safety**: Preventing adverse drug reactions in clinical settings
- **Pharmacology Research**: Understanding how different drugs affect biological pathways

## Models

### GCN (Graph Convolutional Network)
- Two-layer GCN with ReLU activation
- Hidden dimension: 128
- Batch normalization after each layer
- Dropout (p=0.3)
- Learnable node embeddings

### GraphSAGE
- Two-layer GraphSAGE with ReLU activation
- Hidden dimension: 128
- Batch normalization after each layer
- Dropout (p=0.3)
- Learnable node embeddings

### GraphTransformer
- Two-layer Transformer-based GNN with 2 attention heads
- Hidden dimension: 128
- Batch normalization after each layer
- Dropout (p=0.3)
- Learnable node embeddings

## Training

- **Epochs**: Up to 200 per model (with early stopping)
- **Early Stopping**: Patience of 20 evaluation steps
- **Learning Rate**: 0.001
- **Optimizer**: AdamW with weight decay (1e-4)
- **Loss Function**: Binary cross-entropy with logits (numerically stable)
- **Negative Sampling**: Dynamic negative sampling during training
- **Gradient Clipping**: Max norm of 1.0 for stability

## Evaluation

Models are evaluated using the **Hits@20** metric, which measures the percentage of positive edges ranked in the top 20 predictions.

## Requirements

The project uses:
- PyTorch
- PyTorch Geometric
- OGB (Open Graph Benchmark)

See `pixi.toml` for dependency management.

## Usage

### Full Training

Run the main script (trains all 3 models with early stopping):

```bash
python 224w_project.py
```

Or use the advanced trainer with more options:

```bash
# Train GraphSAGE with custom parameters
python trainer.py --model sage --hidden_dim 256 --epochs 200 --patience 20

# Train GCN with different settings
python trainer.py --model gcn --hidden_dim 128 --num_layers 3 --dropout 0.4
```

### Results

Results will be:
- Printed to the console with validation and test metrics
- Logged to `logs/results_YYYYMMDD_HHMMSS.log`

## Output

The script outputs:
- Training progress (loss every 5 epochs)
- Final Hits@20 scores for each model
- Comparison of all three models

Example output:
```
GCN Hits@20       = X.XXXX
GraphSAGE Hits@20 = X.XXXX
Transformer Hits@20 = X.XXXX
```

