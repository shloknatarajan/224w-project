import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, TransformerConv
from torch_geometric.utils import negative_sampling, dropout_edge
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import logging
from datetime import datetime
import os

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

evaluator = Evaluator(name='ogbl-ddi')

class BaseModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Simplified 2-layer decoder with higher dropout to reduce overfitting
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased from 0.2
            nn.Linear(hidden_dim, 1)
        )

    def decode(self, z, edge):
        src, dst = edge[:, 0], edge[:, 1]
        # Concatenate source and destination embeddings
        edge_emb = torch.cat([z[src], z[dst]], dim=1)
        return self.decoder(edge_emb).squeeze()

class GCN(BaseModel):
    def __init__(self, num_nodes, hidden_dim, num_layers=3, dropout=0.5):
        super().__init__(hidden_dim)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

        # Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # Multiple GCN layers with residual connections
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def encode(self, edge_index):
        x = self.emb.weight

        # Apply edge dropout during training for better regularization
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=0.2, training=self.training)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_prev = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Residual connection (skip first layer)
            if i > 0:
                x = x + x_prev

        return x

class GraphSAGE(BaseModel):
    def __init__(self, num_nodes, hidden_dim, num_layers=3, dropout=0.5):
        super().__init__(hidden_dim)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

        # Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # Multiple SAGE layers with residual connections
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def encode(self, edge_index):
        x = self.emb.weight

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_prev = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Residual connection (skip first layer)
            if i > 0:
                x = x + x_prev

        return x

class GraphTransformer(BaseModel):
    def __init__(self, num_nodes, hidden_dim, num_layers=3, heads=4, dropout=0.5):
        super().__init__(hidden_dim)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

        # Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # Multiple Transformer layers with residual connections
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim // heads, heads=heads))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def encode(self, edge_index):
        x = self.emb.weight

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_prev = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Residual connection (skip first layer)
            if i > 0:
                x = x + x_prev

        return x

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
        batch_size = 100000
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

def train_model(name, model, epochs=200, lr=0.01, patience=20, eval_every=5):
    """
    Train model with early stopping and validation.

    Args:
        name: Model name for logging
        model: Model to train
        epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience (number of eval steps without improvement)
        eval_every: Evaluate every N epochs
    """
    logger.info(f"Starting training for {name} (epochs={epochs}, lr={lr}, patience={patience})")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate scheduler - reduce LR when validation plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True  # More aggressive LR reduction
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

        # Reduced negative sampling for memory efficiency
        neg_samples = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=num_nodes,
            num_neg_samples=train_pos.size(0)  # 1:1 ratio
        ).t().to(device)

        # MEMORY-EFFICIENT BATCH DECODING - process in smaller chunks
        batch_size = 100000  # Process edges in chunks
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
            logger.info(
                f"{name} Epoch {epoch:04d}/{epochs} | "
                f"Loss: {loss.item():.4f} | "
                f"Val Hits@20: {val_hits:.4f} | "
                f"Test Hits@20: {test_hits:.4f} | "
                f"Best Val: {best_val_hits:.4f} (epoch {best_epoch}) | "
                f"LR: {current_lr:.6f}"
            )

            # Early stopping check
            if epochs_no_improve >= patience:
                logger.info(f"{name}: Early stopping at epoch {epoch} (no improvement for {patience} eval steps)")
                break

    logger.info(f"{name} FINAL: Best Val Hits@20 = {best_val_hits:.4f} | Test Hits@20 = {best_test_hits:.4f} (at epoch {best_epoch})")
    return best_val_hits, best_test_hits

# OPTIMIZED FOR GCN: Focusing on best-performing model
HIDDEN_DIM = 256  # Increased from 128 for more capacity
NUM_LAYERS = 4    # Increased from 3 for deeper multi-hop learning
DROPOUT = 0.4     # Slightly reduced from 0.5 to allow more capacity while preventing overfitting
EPOCHS = 500      # More epochs since model was still improving at 300
PATIENCE = 40     # Increased patience for deeper model

logger.info("=" * 60)
logger.info(f"Training GCN Model (hidden_dim={HIDDEN_DIM}, layers={NUM_LAYERS})")
logger.info("=" * 60)
gcn_val, gcn_test = train_model(
    "GCN",
    GCN(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT),
    epochs=EPOCHS,
    lr=0.01,
    patience=PATIENCE
)

# COMMENTED OUT: Focusing on GCN only for optimization
# logger.info("\n" + "=" * 60)
# logger.info(f"Training GraphSAGE Model (hidden_dim={HIDDEN_DIM}, layers={NUM_LAYERS})")
# logger.info("=" * 60)
# sage_val, sage_test = train_model(
#     "GraphSAGE",
#     GraphSAGE(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT),
#     epochs=EPOCHS,
#     lr=0.01,
#     patience=PATIENCE
# )
#
# logger.info("\n" + "=" * 60)
# logger.info(f"Training GraphTransformer Model (hidden_dim={HIDDEN_DIM}, layers={NUM_LAYERS})")
# logger.info("=" * 60)
# gt_val, gt_test = train_model(
#     "GraphTransformer",
#     GraphTransformer(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, heads=4, dropout=DROPOUT),
#     epochs=EPOCHS,
#     lr=0.005,  # Lower LR for transformer
#     patience=PATIENCE
# )

logger.info("\n" + "=" * 80)
logger.info("FINAL RESULTS - OPTIMIZED GCN MODEL")
logger.info("=" * 80)
logger.info(f"Model: GCN")
logger.info(f"  Hidden Dimension: {HIDDEN_DIM}")
logger.info(f"  Number of Layers: {NUM_LAYERS}")
logger.info(f"  Dropout: {DROPOUT}")
logger.info(f"  Validation Hits@20: {gcn_val:.4f}")
logger.info(f"  Test Hits@20: {gcn_test:.4f}")
logger.info("=" * 80)

logger.info("\n" + "=" * 80)
logger.info("Training and evaluation completed successfully!")
logger.info(f"Results logged to: {log_filename}")
logger.info("=" * 80)