import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, TransformerConv, GATConv
from torch_geometric.utils import negative_sampling, dropout_edge, add_self_loops
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import logging
from datetime import datetime
import os
import numpy as np

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Setup logging with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'logs/results_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
logger.info(f"Logging results to: {log_filename}")

# -------------------------------
# Load Dataset (NO LEAKAGE)
# -------------------------------
logger.info("Loading dataset ogbl-ddi...")
dataset = PygLinkPropPredDataset('ogbl-ddi')
data = dataset[0]

split_edge = dataset.get_edge_split()
logger.info(f"Dataset loaded: {data.num_nodes} nodes")

train_pos = split_edge['train']['edge'].to(device)
valid_pos = split_edge['valid']['edge'].to(device)
valid_neg = split_edge['valid']['edge_neg'].to(device)  # Official negatives!
test_pos  = split_edge['test']['edge'].to(device)
test_neg  = split_edge['test']['edge_neg'].to(device)   # Official negatives!

logger.info(f"Train pos edges: {train_pos.size(0)}, Valid pos: {valid_pos.size(0)}, Test pos: {test_pos.size(0)}")
logger.info(f"Valid neg edges: {valid_neg.size(0)}, Test neg: {test_neg.size(0)}")

num_nodes = data.num_nodes

# Construct graph using *only* training edges (IMPORTANT: prevents leakage)
data.edge_index = train_pos.t().contiguous().to(device)

# Add self-loops for better feature aggregation
data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=num_nodes)
logger.info(f"Added self-loops: Total edges now = {data.edge_index.size(1)}")

# Compute node degrees as structural features
from torch_geometric.utils import degree
node_degrees = degree(data.edge_index[0], num_nodes=num_nodes, dtype=torch.float).to(device)
# Normalize and add small constant to avoid log(0)
node_degree_features = torch.log(node_degrees + 1).unsqueeze(1)  # [num_nodes, 1]
logger.info(f"Computed node degree features: mean={node_degree_features.mean():.2f}, std={node_degree_features.std():.2f}")

evaluator = Evaluator(name='ogbl-ddi')

class ImprovedEdgeDecoder(nn.Module):
    """Simplified edge decoder to reduce overfitting"""
    def __init__(self, hidden_dim, dropout=0.5):
        super().__init__()
        # Simpler decoder: just Hadamard product + 1-layer MLP
        # Input: hadamard = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z, edge):
        src, dst = edge[:, 0], edge[:, 1]
        z_src, z_dst = z[src], z[dst]

        # Hadamard (element-wise) product - captures feature interactions
        hadamard = z_src * z_dst

        # Simple MLP scoring
        score = self.mlp(hadamard).squeeze()

        return score

class BaseModel(nn.Module):
    def __init__(self, hidden_dim, decoder_dropout=0.3):
        super().__init__()
        self.decoder = ImprovedEdgeDecoder(hidden_dim, dropout=decoder_dropout)

    def decode(self, z, edge):
        return self.decoder(z, edge)

class GCN(BaseModel):
    def __init__(self, num_nodes, hidden_dim, num_layers=3, dropout=0.4, decoder_dropout=0.3, use_degree_features=True):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_degree_features = use_degree_features

        # Learnable embeddings with improved initialization
        self.emb = nn.Embedding(num_nodes, hidden_dim - 1 if use_degree_features else hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # Multiple GCN layers with residual connections - keep BatchNorm as it worked in baseline
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))  # We already added self-loops
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def encode(self, edge_index):
        x = self.emb.weight

        # Concatenate degree features with learnable embeddings
        if self.use_degree_features:
            x = torch.cat([x, node_degree_features], dim=1)

        # Moderate edge dropout for regularization - only during training
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=0.2, training=self.training)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_prev = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Residual connection with scaling (for all layers except first)
            if i > 0:
                x = x + 0.5 * x_prev  # Scaled residual for better gradient flow

        return x

class GraphSAGE(BaseModel):
    def __init__(self, num_nodes, hidden_dim, num_layers=3, dropout=0.5, use_jk=True):
        super().__init__(hidden_dim)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_jk = use_jk

        # Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # Multiple SAGE layers with layer normalization
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.lns.append(nn.LayerNorm(hidden_dim))

        # Jump Knowledge
        if use_jk:
            self.jk_proj = nn.Linear(hidden_dim * (num_layers + 1), hidden_dim)

    def encode(self, edge_index):
        x = self.emb.weight
        xs = [x]

        # Apply edge dropout during training
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=0.2, training=self.training)

        for i, (conv, ln) in enumerate(zip(self.convs, self.lns)):
            x_prev = x
            x = conv(x, edge_index)
            x = ln(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection (skip first layer)
            if i > 0:
                x = x + x_prev

            xs.append(x)

        # Jump Knowledge
        if self.use_jk:
            x = torch.cat(xs, dim=1)
            x = self.jk_proj(x)

        return x

class GraphTransformer(BaseModel):
    def __init__(self, num_nodes, hidden_dim, num_layers=3, heads=4, dropout=0.5, use_jk=True):
        super().__init__(hidden_dim)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_jk = use_jk

        # Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # Multiple Transformer layers with layer normalization
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim // heads, heads=heads))
            self.lns.append(nn.LayerNorm(hidden_dim))

        # Jump Knowledge
        if use_jk:
            self.jk_proj = nn.Linear(hidden_dim * (num_layers + 1), hidden_dim)

    def encode(self, edge_index):
        x = self.emb.weight
        xs = [x]

        # Apply edge dropout during training
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=0.2, training=self.training)

        for i, (conv, ln) in enumerate(zip(self.convs, self.lns)):
            x_prev = x
            x = conv(x, edge_index)
            x = ln(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection (skip first layer)
            if i > 0:
                x = x + x_prev

            xs.append(x)

        # Jump Knowledge
        if self.use_jk:
            x = torch.cat(xs, dim=1)
            x = self.jk_proj(x)

        return x

class GAT(BaseModel):
    """Graph Attention Network - often performs well on link prediction"""
    def __init__(self, num_nodes, hidden_dim, num_layers=3, heads=4, dropout=0.5, use_jk=True):
        super().__init__(hidden_dim)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_jk = use_jk
        self.heads = heads

        # Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # Multiple GAT layers
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer: average attention heads
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=0.2))
            else:
                # Hidden layers: concatenate attention heads
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, dropout=0.2))
            self.lns.append(nn.LayerNorm(hidden_dim))

        # Jump Knowledge
        if use_jk:
            self.jk_proj = nn.Linear(hidden_dim * (num_layers + 1), hidden_dim)

    def encode(self, edge_index):
        x = self.emb.weight
        xs = [x]

        # Apply edge dropout during training
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=0.2, training=self.training)

        for i, (conv, ln) in enumerate(zip(self.convs, self.lns)):
            x_prev = x
            x = conv(x, edge_index)
            x = ln(x)
            x = F.elu(x)  # ELU works better with GAT
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection (skip first layer)
            if i > 0:
                x = x + x_prev

            xs.append(x)

        # Jump Knowledge
        if self.use_jk:
            x = torch.cat(xs, dim=1)
            x = self.jk_proj(x)

        return x

def hard_negative_mining(model, z, edge_index, num_samples, top_k_ratio=0.3):
    """
    Sample hard negatives - negatives with high predicted scores that are challenging for the model.

    Args:
        model: The model
        z: Node embeddings
        edge_index: Current edge index
        num_samples: Number of negatives to return
        top_k_ratio: Ratio of hard negatives to mine from a larger pool
    """
    # Sample more negatives than needed
    sample_size = int(num_samples / top_k_ratio)
    neg_edges = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=sample_size
    ).t().to(device)

    # Score all negative samples
    with torch.no_grad():
        neg_scores = model.decode(z, neg_edges)

    # Select top-k hardest negatives (highest scores = closest to positive)
    _, indices = torch.topk(neg_scores, k=num_samples)
    hard_negatives = neg_edges[indices]

    return hard_negatives

def get_loss(model, edge_index, pos_edges, emb_reg_weight=0.001):
    z = model.encode(edge_index)
    pos_score = model.decode(z, pos_edges)

    neg_edges = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=pos_edges.size(0)
    ).to(device)

    neg_score = model.decode(z, neg_edges)

    # BCE loss (more numerically stable)
    loss = -torch.log(torch.sigmoid(pos_score)).mean() - torch.log(1 - torch.sigmoid(neg_score)).mean()

    # Add L2 regularization on embeddings to reduce overfitting
    emb_reg_loss = emb_reg_weight * torch.norm(model.emb.weight, p=2)
    loss = loss + emb_reg_loss

    return loss

def evaluate(model, pos_edges, neg_edges):
    """Evaluate model using official OGB negative edges for consistent evaluation."""
    model.eval()
    with torch.no_grad():
        z = model.encode(data.edge_index)

        # Positive scores - batch process to avoid OOM
        pos_scores_list = []
        batch_size = 200000  # Larger batches since we use smaller model
        for i in range(0, pos_edges.size(0), batch_size):
            chunk = pos_edges[i:i+batch_size]
            pos_scores_list.append(model.decode(z, chunk).view(-1).cpu())
        pos_scores = torch.cat(pos_scores_list)

        # Negative scores - batch process to avoid OOM
        neg_scores_list = []
        for i in range(0, neg_edges.size(0), batch_size):
            chunk = neg_edges[i:i+batch_size]
            neg_scores_list.append(model.decode(z, chunk).view(-1).cpu())
        neg_scores = torch.cat(neg_scores_list)

        # Use OGB evaluator with official negative samples
        result = evaluator.eval({
            'y_pred_pos': pos_scores,
            'y_pred_neg': neg_scores,
        })

        return result['hits@20']

def train_model(name, model, epochs=200, lr=0.01, patience=20, eval_every=5, use_hard_negatives=True, hard_neg_ratio=0.3):
    """
    Train model with early stopping, validation, and hard negative mining.

    Args:
        name: Model name for logging
        model: Model to train
        epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience (number of eval steps without improvement)
        eval_every: Evaluate every N epochs
        use_hard_negatives: Whether to use hard negative mining
        hard_neg_ratio: Ratio of hard negatives to use (rest are random)
    """
    logger.info(f"Starting training for {name} (epochs={epochs}, lr={lr}, patience={patience}, hard_neg={use_hard_negatives})")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler - reduce LR when validation plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True  # Less aggressive LR reduction
    )

    best_val_hits = 0
    best_test_hits = 0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Encode once per epoch - keep in memory
        z = model.encode(data.edge_index)

        # Mix of random and hard negatives (after initial warmup)
        num_negatives = train_pos.size(0)
        warmup_epochs = 10  # Shorter warmup for hard negatives
        if use_hard_negatives and epoch > warmup_epochs:  # Warmup period before hard negatives
            num_hard = int(num_negatives * hard_neg_ratio)
            num_random = num_negatives - num_hard

            # Hard negatives
            hard_neg = hard_negative_mining(model, z, data.edge_index, num_hard, top_k_ratio=0.3)

            # Random negatives
            random_neg = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=num_nodes,
                num_neg_samples=num_random
            ).t().to(device)

            # Combine
            neg_samples = torch.cat([hard_neg, random_neg], dim=0)
        else:
            # Pure random negative sampling (for warmup or if hard negatives disabled)
            neg_samples = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=num_nodes,
                num_neg_samples=num_negatives
            ).t().to(device)

        # MEMORY-EFFICIENT BATCH DECODING - process in smaller chunks
        batch_size = 100000  # Larger batches since we use smaller hidden_dim
        pos_out_list = []
        for i in range(0, train_pos.size(0), batch_size):
            chunk = train_pos[i:i+batch_size]
            pos_out_list.append(model.decode(z, chunk))
        pos_out = torch.cat(pos_out_list)

        neg_out_list = []
        for i in range(0, neg_samples.size(0), batch_size):
            chunk = neg_samples[i:i+batch_size]
            neg_out_list.append(model.decode(z, chunk))
        neg_out = torch.cat(neg_out_list)

        # IMPROVED LOSS: Use BCE with logits (more numerically stable)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_out, torch.ones_like(pos_out)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_out, torch.zeros_like(neg_out)
        )
        loss = pos_loss + neg_loss

        # Add L2 regularization on embeddings to reduce overfitting (CRITICAL!)
        emb_reg_weight = 0.01  # Strong regularization
        emb_reg_loss = emb_reg_weight * torch.norm(model.emb.weight, p=2)
        loss = loss + emb_reg_loss

        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Clear intermediate tensors
        del z, neg_samples, pos_out, neg_out, pos_loss, neg_loss
        torch.cuda.empty_cache()

        # EARLY STOPPING: Evaluate periodically using official negatives
        if epoch % eval_every == 0 or epoch == 1:
            val_hits = evaluate(model, valid_pos, valid_neg)
            test_hits = evaluate(model, test_pos, test_neg)
            
            # Free up memory after evaluation
            torch.cuda.empty_cache()

            # Update learning rate based on validation performance
            scheduler.step(val_hits)

            improved = val_hits > best_val_hits
            if improved:
                best_val_hits = val_hits
                best_test_hits = test_hits
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            current_lr = optimizer.param_groups[0]['lr']
            improvement_marker = "🔥" if improved else ""
            hard_neg_status = f"[Hard Neg]" if use_hard_negatives and epoch > warmup_epochs else "[Random Neg]"
            logger.info(
                f"{name} Epoch {epoch:04d}/{epochs} {hard_neg_status} | "
                f"Loss: {loss.item():.4f} | "
                f"Val Hits@20: {val_hits:.4f} | "
                f"Test Hits@20: {test_hits:.4f} | "
                f"Best Val: {best_val_hits:.4f} (epoch {best_epoch}) | "
                f"LR: {current_lr:.6f} {improvement_marker}"
            )

            # Early stopping check
            if epochs_no_improve >= patience:
                logger.info(f"{name}: Early stopping at epoch {epoch} (no improvement for {patience} eval steps)")
                break

    logger.info(f"{name} FINAL: Best Val Hits@20 = {best_val_hits:.4f} | Test Hits@20 = {best_test_hits:.4f} (at epoch {best_epoch})")
    return best_val_hits, best_test_hits

# ENHANCED HYPERPARAMETERS: Tuned to reduce overfitting
HIDDEN_DIM = 128  # Memory-efficient (256 causes OOM)
NUM_LAYERS = 3    # 3 layers for good expressiveness (4 too memory intensive)
DROPOUT = 0.6     # INCREASED dropout for stronger regularization
DECODER_DROPOUT = 0.5  # INCREASED decoder dropout
EPOCHS = 300      # Reduced epochs with better early stopping
PATIENCE = 30     # More aggressive early stopping
HEADS = 4         # Number of attention heads for GAT/Transformer (reduced from 8)
LEARNING_RATE = 0.005  # REDUCED learning rate for more stable training
WEIGHT_DECAY = 1e-4  # INCREASED weight decay for regularization
EDGE_DROPOUT = 0.2  # INCREASED edge dropout probability
USE_HARD_NEGATIVES = True  # ENABLED for better generalization
HARD_NEG_RATIO = 0.3  # Ratio of hard negatives
HARD_NEG_WARMUP = 10  # Start hard negatives earlier

# Consolidated configuration dictionary for easy logging and tracking
config = {
    'model_name': 'GCN-Enhanced-v3-AntiOverfit',
    'hidden_dim': HIDDEN_DIM,
    'num_layers': NUM_LAYERS,
    'dropout': DROPOUT,
    'decoder_dropout': DECODER_DROPOUT,
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'heads': HEADS,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'edge_dropout': EDGE_DROPOUT,
    'emb_reg_weight': 0.01,  # UPDATED: Strong embedding regularization
    'batch_size': 100000,
    'eval_batch_size': 200000,
    'eval_every': 5,
    'scheduler_factor': 0.5,
    'scheduler_patience': 10,
    'gradient_clip_max_norm': 1.0,
    'use_hard_negatives': USE_HARD_NEGATIVES,
    'hard_neg_ratio': HARD_NEG_RATIO,
    'hard_neg_warmup': HARD_NEG_WARMUP,
    'use_self_loops': True,
    'scaled_residual': True,
    'use_degree_features': True,  # NEW: Node degree features
    'decoder_type': 'simplified_hadamard',  # UPDATED: Simplified decoder
    'num_nodes': num_nodes,
    'train_edges': train_pos.size(0),
    'val_edges': valid_pos.size(0),
    'test_edges': test_pos.size(0)
}

logger.info("=" * 80)
logger.info(f"MODEL CONFIGURATION - {config['model_name']}")
logger.info("=" * 80)
logger.info("Architecture:")
logger.info(f"  Model: {config['model_name']}")
logger.info(f"  Decoder: {config['decoder_type']} (simplified 2-layer MLP)")
logger.info(f"  Hidden Dim: {config['hidden_dim']}, Layers: {config['num_layers']}")
logger.info(f"  Dropout: {config['dropout']}, Decoder Dropout: {config['decoder_dropout']}")
logger.info(f"  Self-loops: {config['use_self_loops']}, Scaled Residual: {config['scaled_residual']}")
logger.info(f"  Node Features: Degree features (log-normalized)")
logger.info("")
logger.info("Training:")
logger.info(f"  Epochs: {config['epochs']}, Patience: {config['patience']}")
logger.info(f"  Learning Rate: {config['learning_rate']}, Weight Decay: {config['weight_decay']}")
logger.info(f"  Edge Dropout: {config['edge_dropout']}, Embedding L2: {config['emb_reg_weight']}")
logger.info(f"  Hard Negatives: {config['use_hard_negatives']} (ratio={config['hard_neg_ratio']}, warmup={config['hard_neg_warmup']})")
logger.info(f"  Scheduler: factor={config['scheduler_factor']}, patience={config['scheduler_patience']}")
logger.info("")
logger.info("Dataset:")
logger.info(f"  Nodes: {config['num_nodes']}")
logger.info(f"  Train Edges: {config['train_edges']}")
logger.info(f"  Val Edges: {config['val_edges']}, Test Edges: {config['test_edges']}")
logger.info("=" * 80)

gcn_val, gcn_test = train_model(
    config['model_name'],
    GCN(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, decoder_dropout=DECODER_DROPOUT),
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    patience=PATIENCE,
    use_hard_negatives=USE_HARD_NEGATIVES,
    hard_neg_ratio=HARD_NEG_RATIO
)

# Comment out other models for now to focus on one model first
# logger.info("\n" + "=" * 60)
# logger.info(f"Training GraphSAGE Model (hidden_dim={HIDDEN_DIM}, layers={NUM_LAYERS})")
# logger.info("=" * 60)
# sage_val, sage_test = train_model(
#     "GraphSAGE",
#     GraphSAGE(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, use_jk=False),
#     epochs=EPOCHS,
#     lr=0.01,
#     patience=PATIENCE
# )

# logger.info("\n" + "=" * 60)
# logger.info(f"Training GAT Model (hidden_dim={HIDDEN_DIM}, layers={NUM_LAYERS}, heads={HEADS})")
# logger.info("=" * 60)
# gat_val, gat_test = train_model(
#     "GAT",
#     GAT(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, heads=HEADS, dropout=DROPOUT, use_jk=False),
#     epochs=EPOCHS,
#     lr=0.005,  # Lower LR for attention models
#     patience=PATIENCE
# )

logger.info("\n" + "=" * 80)
logger.info("FINAL RESULTS")
logger.info("=" * 80)
logger.info(f"Model: {config['model_name']}")
logger.info(f"Configuration: Hidden Dim={config['hidden_dim']}, Layers={config['num_layers']}, Dropout={config['dropout']}")
logger.info("")
logger.info(f"{config['model_name']}:")
logger.info(f"  Validation Hits@20: {gcn_val:.4f}")
logger.info(f"  Test Hits@20: {gcn_test:.4f}")
val_test_gap = gcn_val - gcn_test
logger.info(f"  Val-Test Gap: {val_test_gap:.4f} ({val_test_gap/gcn_val*100:.1f}% relative)")
logger.info("=" * 80)

logger.info("\n" + "=" * 80)
logger.info("Anti-Overfitting Improvements (v3)")
logger.info("=" * 80)
logger.info("1. Simplified Decoder Architecture:")
logger.info("   - Reduced from 3-layer to 2-layer MLP (less capacity)")
logger.info("   - Simple Hadamard product scoring (no complex multi-strategy)")
logger.info("   - Increased decoder dropout: 0.3 → 0.5")
logger.info("")
logger.info("2. Added Structural Features:")
logger.info("   - Node degree features (log-normalized)")
logger.info("   - Provides inductive bias for link prediction")
logger.info("   - Reduces reliance on pure memorization")
logger.info("")
logger.info("3. Strong Regularization:")
logger.info(f"   - Encoder dropout: 0.4 → {config['dropout']}")
logger.info(f"   - Edge dropout: 0.15 → {config['edge_dropout']}")
logger.info(f"   - Embedding L2 regularization: {config['emb_reg_weight']} (NEWLY APPLIED!)")
logger.info(f"   - Weight decay: 5e-5 → {config['weight_decay']}")
logger.info("")
logger.info("4. Improved Training Dynamics:")
logger.info(f"   - Learning rate: 0.01 → {config['learning_rate']} (slower, more stable)")
logger.info(f"   - Hard negative mining ENABLED (ratio={config['hard_neg_ratio']}, warmup={config['hard_neg_warmup']})")
logger.info(f"   - More aggressive early stopping (patience={config['patience']})")
logger.info("=" * 80)

logger.info("\n" + "=" * 80)
logger.info("Training and evaluation completed successfully!")
logger.info(f"Results logged to: {log_filename}")
logger.info("=" * 80)