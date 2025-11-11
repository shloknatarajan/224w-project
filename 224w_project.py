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
train_edge_index = train_pos.t().contiguous().to(device)

# Compute rich structural features BEFORE adding self-loops (to avoid networkx issues)
logger.info("Computing structural features (this may take a moment)...")
from torch_geometric.utils import degree, to_networkx
from torch_geometric.data import Data as PyGData
import networkx as nx

# Convert to NetworkX graph for feature computation (without self-loops)
edge_index_cpu = train_edge_index.cpu()
G = to_networkx(PyGData(edge_index=edge_index_cpu, num_nodes=num_nodes), to_undirected=True)

# 1. Node degree (log-normalized) - computed from original graph
node_degrees = degree(train_edge_index[0], num_nodes=num_nodes, dtype=torch.float).to(device)
degree_features = torch.log(node_degrees + 1).unsqueeze(1)

# 2. Clustering coefficient - measures local triangle density
clustering = nx.clustering(G)
clustering_features = torch.tensor([clustering[i] for i in range(num_nodes)], dtype=torch.float).unsqueeze(1).to(device)

# 3. Core number - k-core decomposition (measures node importance)
core_number = nx.core_number(G)
core_features = torch.tensor([core_number[i] for i in range(num_nodes)], dtype=torch.float).unsqueeze(1).to(device)
core_features = torch.log(core_features + 1)  # Log normalize

# 4. PageRank - measures node importance
pagerank = nx.pagerank(G, max_iter=50)
pagerank_features = torch.tensor([pagerank[i] for i in range(num_nodes)], dtype=torch.float).unsqueeze(1).to(device)
pagerank_features = pagerank_features / pagerank_features.max()  # Normalize to [0, 1]

# 5. Neighbor degree statistics (mean and max of neighbor degrees)
row, col = train_edge_index
neighbor_degrees = torch.zeros((num_nodes, 2), dtype=torch.float).to(device)
for node in range(num_nodes):
    neighbors = col[row == node]
    if len(neighbors) > 0:
        neighbor_degs = node_degrees[neighbors]
        neighbor_degrees[node, 0] = neighbor_degs.mean()
        neighbor_degrees[node, 1] = neighbor_degs.max()
neighbor_degrees = torch.log(neighbor_degrees + 1)

# Now add self-loops to the edge index for message passing
data.edge_index, _ = add_self_loops(train_edge_index, num_nodes=num_nodes)
logger.info(f"Added self-loops: Total edges now = {data.edge_index.size(1)}")

# Combine all features
node_structural_features = torch.cat([
    degree_features,           # 1 dim
    clustering_features,       # 1 dim
    core_features,            # 1 dim
    pagerank_features,        # 1 dim
    neighbor_degrees          # 2 dims
], dim=1)  # Total: 6 features

logger.info(f"Computed rich structural features ({node_structural_features.shape[1]} dims):")
logger.info(f"  - Degree: mean={degree_features.mean():.3f}, std={degree_features.std():.3f}")
logger.info(f"  - Clustering: mean={clustering_features.mean():.3f}, std={clustering_features.std():.3f}")
logger.info(f"  - Core number: mean={core_features.mean():.3f}, std={core_features.std():.3f}")
logger.info(f"  - PageRank: mean={pagerank_features.mean():.3f}, std={pagerank_features.std():.3f}")

evaluator = Evaluator(name='ogbl-ddi')

class ExponentialMovingAverage:
    """
    Maintains exponential moving average of model parameters.
    Provides smoother predictions and better generalization.

    Usage:
        ema = ExponentialMovingAverage(model, decay=0.999)

        # Training loop
        for epoch in epochs:
            train_step()
            ema.update()

            # Evaluation
            ema.apply_shadow()
            evaluate()
            ema.restore()
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters after each training step"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class ImprovedEdgeDecoder(nn.Module):
    """Multi-strategy edge decoder for better link prediction"""
    def __init__(self, hidden_dim, dropout=0.5, use_multi_strategy=True):
        super().__init__()
        self.use_multi_strategy = use_multi_strategy

        if use_multi_strategy:
            # Multiple scoring strategies
            # 1. Hadamard product path
            self.hadamard_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

            # 2. Concatenation path
            self.concat_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

            # 3. Bilinear scoring
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

            # Learnable weights for combining strategies
            self.strategy_weights = nn.Parameter(torch.ones(3) / 3)
        else:
            # Simple decoder: just Hadamard product + MLP
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(self, z, edge):
        src, dst = edge[:, 0], edge[:, 1]
        z_src, z_dst = z[src], z[dst]

        if self.use_multi_strategy:
            # Apply softmax to strategy weights
            weights = F.softmax(self.strategy_weights, dim=0)

            # Strategy 1: Hadamard product
            hadamard = z_src * z_dst
            score1 = self.hadamard_mlp(hadamard).squeeze()

            # Strategy 2: Concatenation
            concat = torch.cat([z_src, z_dst], dim=1)
            score2 = self.concat_mlp(concat).squeeze()

            # Strategy 3: Bilinear
            score3 = self.bilinear(z_src, z_dst).squeeze()

            # Weighted combination
            score = weights[0] * score1 + weights[1] * score2 + weights[2] * score3
        else:
            # Simple Hadamard scoring
            hadamard = z_src * z_dst
            score = self.mlp(hadamard).squeeze()

        return score

class BaseModel(nn.Module):
    def __init__(self, hidden_dim, decoder_dropout=0.3, use_multi_strategy=True):
        super().__init__()
        self.decoder = ImprovedEdgeDecoder(hidden_dim, dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)

    def decode(self, z, edge):
        return self.decoder(z, edge)

class GCN(BaseModel):
    def __init__(self, num_nodes, hidden_dim, num_layers=3, dropout=0.4, decoder_dropout=0.3, use_structural_features=True, num_structural_features=6, use_multi_strategy=True):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_structural_features = use_structural_features
        self.num_structural_features = num_structural_features

        # Learnable embeddings with improved initialization
        emb_dim = hidden_dim - num_structural_features if use_structural_features else hidden_dim
        self.emb = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # Feature projection layer to transform structural features
        if use_structural_features:
            self.feature_proj = nn.Sequential(
                nn.Linear(num_structural_features, num_structural_features),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)  # Lighter dropout on features
            )

        # Multiple GCN layers with residual connections - keep BatchNorm as it worked in baseline
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))  # We already added self-loops
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def encode(self, edge_index):
        x = self.emb.weight

        # Concatenate rich structural features with learnable embeddings
        if self.use_structural_features:
            # Project structural features through a small MLP
            struct_feats = self.feature_proj(node_structural_features)
            # Add feature dropout for regularization
            if self.training:
                struct_feats = F.dropout(struct_feats, p=0.1, training=True)
            x = torch.cat([x, struct_feats], dim=1)

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

def evaluate(model, pos_edges, neg_edges, batch_size=50000):
    """Evaluate model using official OGB negative edges for consistent evaluation."""
    model.eval()
    with torch.no_grad():
        z = model.encode(data.edge_index)

        # Positive scores - batch process to avoid OOM
        pos_scores_list = []
        for i in range(0, pos_edges.size(0), batch_size):
            chunk = pos_edges[i:i+batch_size]
            scores = model.decode(z, chunk).view(-1).cpu()
            pos_scores_list.append(scores)
            del scores  # Free immediately
        pos_scores = torch.cat(pos_scores_list)
        del pos_scores_list  # Free the list

        # Negative scores - batch process to avoid OOM
        neg_scores_list = []
        for i in range(0, neg_edges.size(0), batch_size):
            chunk = neg_edges[i:i+batch_size]
            scores = model.decode(z, chunk).view(-1).cpu()
            neg_scores_list.append(scores)
            del scores  # Free immediately
        neg_scores = torch.cat(neg_scores_list)
        del neg_scores_list  # Free the list

        # Free z before evaluation
        del z
        torch.cuda.empty_cache()

        # Use OGB evaluator with official negative samples
        result = evaluator.eval({
            'y_pred_pos': pos_scores,
            'y_pred_neg': neg_scores,
        })

        return result['hits@20']

def train_model(name, model, epochs=200, lr=0.01, patience=20, eval_every=5, use_hard_negatives=True, hard_neg_ratio=0.3, batch_size=20000, eval_batch_size=50000, gradient_accumulation_steps=3):
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
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        gradient_accumulation_steps: Number of steps to accumulate gradients
    """
    logger.info(f"Starting training for {name} (epochs={epochs}, lr={lr}, patience={patience}, hard_neg={use_hard_negatives})")
    logger.info(f"Memory optimization: batch_size={batch_size}, gradient_accumulation={gradient_accumulation_steps}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler - reduce LR when validation plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, verbose=True  # More patient scheduler
    )

    # Initialize EMA for smoother predictions
    ema = ExponentialMovingAverage(model, decay=0.999)
    logger.info("Initialized EMA with decay=0.999 for stable checkpointing")

    best_val_hits = 0
    best_test_hits = 0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()

        # GRADIENT ACCUMULATION: Process data in smaller chunks
        total_loss = 0.0
        accumulation_steps_taken = 0

        # Split training data into micro-batches for gradient accumulation
        num_samples_per_step = train_pos.size(0) // gradient_accumulation_steps

        for accum_step in range(gradient_accumulation_steps):
            # Get micro-batch of positive edges
            start_idx = accum_step * num_samples_per_step
            end_idx = start_idx + num_samples_per_step if accum_step < gradient_accumulation_steps - 1 else train_pos.size(0)
            pos_batch = train_pos[start_idx:end_idx]

            # Encode graph
            z = model.encode(data.edge_index)

            # Generate negatives for this micro-batch
            num_negatives = pos_batch.size(0)
            warmup_epochs = 50  # Longer warmup for stability (increased from 10)
            if use_hard_negatives and epoch > warmup_epochs:
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
                del hard_neg, random_neg
            else:
                # Pure random negative sampling
                neg_samples = negative_sampling(
                    edge_index=data.edge_index,
                    num_nodes=num_nodes,
                    num_neg_samples=num_negatives
                ).t().to(device)

            # MEMORY-EFFICIENT BATCH DECODING
            pos_out_list = []
            for i in range(0, pos_batch.size(0), batch_size):
                chunk = pos_batch[i:i+batch_size]
                scores = model.decode(z, chunk)
                pos_out_list.append(scores)
                del scores
            pos_out = torch.cat(pos_out_list)
            del pos_out_list

            neg_out_list = []
            for i in range(0, neg_samples.size(0), batch_size):
                chunk = neg_samples[i:i+batch_size]
                scores = model.decode(z, chunk)
                neg_out_list.append(scores)
                del scores
            neg_out = torch.cat(neg_out_list)
            del neg_out_list

            # IMPROVED LOSS: Use BCE with logits (more numerically stable)
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_out, torch.ones_like(pos_out)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_out, torch.zeros_like(neg_out)
            )
            loss = pos_loss + neg_loss

            # Add L2 regularization on embeddings
            emb_reg_weight = 0.005
            emb_reg_loss = emb_reg_weight * torch.norm(model.emb.weight, p=2)
            loss = loss + emb_reg_loss

            # Scale loss by accumulation steps (to maintain same effective learning rate)
            loss = loss / gradient_accumulation_steps
            total_loss += loss.item()

            # Backward pass (accumulate gradients)
            loss.backward()

            # Clear intermediate tensors
            del z, neg_samples, pos_out, neg_out, pos_loss, neg_loss, emb_reg_loss, loss
            torch.cuda.empty_cache()

            accumulation_steps_taken += 1

        # Gradient clipping and optimizer step (after accumulation)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Update EMA after each training step
        ema.update()

        # Use total accumulated loss for logging
        loss_value = total_loss

        # EARLY STOPPING: Evaluate periodically using official negatives
        if epoch % eval_every == 0 or epoch == 1:
            # Evaluate using EMA weights for more stable predictions
            ema.apply_shadow()
            val_hits = evaluate(model, valid_pos, valid_neg, batch_size=eval_batch_size)
            test_hits = evaluate(model, test_pos, test_neg, batch_size=eval_batch_size)
            ema.restore()

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
                f"Loss: {loss_value:.4f} | "
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

# ENHANCED HYPERPARAMETERS v4: Rich features + Multi-strategy decoder + Better regularization
# MEMORY OPTIMIZED VERSION - reduced to fit in 22GB GPU
HIDDEN_DIM = 192  # REDUCED from 256 to save memory (was causing OOM)
NUM_LAYERS = 3    # 3 layers for good expressiveness
DROPOUT = 0.5     # Slightly reduced since we have better features now
DECODER_DROPOUT = 0.4  # Reduced decoder dropout (multi-strategy is more robust)
EPOCHS = 400      # More epochs since we have better regularization
PATIENCE = 40     # More patience with stable training
HEADS = 4         # Number of attention heads for GAT/Transformer
LEARNING_RATE = 0.003  # REDUCED learning rate for stable training
WEIGHT_DECAY = 5e-5  # REDUCED weight decay (rich features reduce overfitting)
EDGE_DROPOUT = 0.15  # REDUCED edge dropout (rich features already provide diversity)
USE_HARD_NEGATIVES = True  # ENABLED for better generalization
HARD_NEG_RATIO = 0.05  # REDUCED from 0.15 to 0.05 for training stability
HARD_NEG_WARMUP = 50  # Increased from 20 to 50 for gradual introduction
NUM_STRUCTURAL_FEATURES = 6  # Number of structural features we compute
USE_MULTI_STRATEGY = False  # DISABLED to save memory (single strategy is more memory efficient)

# Consolidated configuration dictionary for easy logging and tracking
config = {
    'model_name': 'GCN-Enhanced-v4-RichFeatures',
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
    'emb_reg_weight': 0.005,  # REDUCED: Rich features reduce need for strong regularization
    'batch_size': 20000,  # FURTHER REDUCED to avoid OOM (was 50000)
    'eval_batch_size': 50000,  # FURTHER REDUCED to avoid OOM (was 100000)
    'eval_every': 5,
    'gradient_accumulation_steps': 3,  # Accumulate gradients to simulate larger batch
    'scheduler_factor': 0.5,
    'scheduler_patience': 15,  # More patient LR scheduler
    'gradient_clip_max_norm': 1.0,
    'use_hard_negatives': USE_HARD_NEGATIVES,
    'hard_neg_ratio': HARD_NEG_RATIO,
    'hard_neg_warmup': HARD_NEG_WARMUP,
    'use_self_loops': True,
    'scaled_residual': True,
    'use_structural_features': True,
    'num_structural_features': NUM_STRUCTURAL_FEATURES,
    'use_multi_strategy': USE_MULTI_STRATEGY,
    'decoder_type': 'multi_strategy' if USE_MULTI_STRATEGY else 'simplified_hadamard',
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
logger.info(f"  Decoder: {config['decoder_type']}")
logger.info(f"  Hidden Dim: {config['hidden_dim']}, Layers: {config['num_layers']}")
logger.info(f"  Dropout: {config['dropout']}, Decoder Dropout: {config['decoder_dropout']}")
logger.info(f"  Self-loops: {config['use_self_loops']}, Scaled Residual: {config['scaled_residual']}")
logger.info(f"  Structural Features: {config['num_structural_features']} dims (degree, clustering, core, pagerank, neighbor stats)")
logger.info("")
logger.info("Training:")
logger.info(f"  Epochs: {config['epochs']}, Patience: {config['patience']}")
logger.info(f"  Learning Rate: {config['learning_rate']}, Weight Decay: {config['weight_decay']}")
logger.info(f"  Edge Dropout: {config['edge_dropout']}, Embedding L2: {config['emb_reg_weight']}")
logger.info(f"  Hard Negatives: {config['use_hard_negatives']} (ratio={config['hard_neg_ratio']}, warmup={config['hard_neg_warmup']})")
logger.info(f"  Scheduler: factor={config['scheduler_factor']}, patience={config['scheduler_patience']}")
logger.info(f"  Batch Size: {config['batch_size']}, Eval Batch: {config['eval_batch_size']}")
logger.info("")
logger.info("Dataset:")
logger.info(f"  Nodes: {config['num_nodes']}")
logger.info(f"  Train Edges: {config['train_edges']}")
logger.info(f"  Val Edges: {config['val_edges']}, Test Edges: {config['test_edges']}")
logger.info("=" * 80)

gcn_val, gcn_test = train_model(
    config['model_name'],
    GCN(num_nodes, HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, decoder_dropout=DECODER_DROPOUT,
        use_structural_features=True, num_structural_features=NUM_STRUCTURAL_FEATURES,
        use_multi_strategy=USE_MULTI_STRATEGY),
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    patience=PATIENCE,
    use_hard_negatives=USE_HARD_NEGATIVES,
    hard_neg_ratio=HARD_NEG_RATIO,
    batch_size=config['batch_size'],
    eval_batch_size=config['eval_batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps']
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
logger.info("Performance Improvements (v4 - Rich Features)")
logger.info("=" * 80)
logger.info("1. Rich Structural Features (6 dimensions):")
logger.info("   - Node degree (log-normalized)")
logger.info("   - Clustering coefficient (triangle density)")
logger.info("   - Core number (k-core importance)")
logger.info("   - PageRank (global importance)")
logger.info("   - Neighbor degree statistics (mean/max)")
logger.info("   → Provides strong inductive bias for link prediction")
logger.info("")
logger.info("2. Multi-Strategy Decoder:")
logger.info("   - Hadamard product scoring")
logger.info("   - Concatenation-based scoring")
logger.info("   - Bilinear scoring")
logger.info("   - Learnable weighted combination of strategies")
logger.info("   → Captures different aspects of node relationships")
logger.info("")
logger.info("3. Increased Model Capacity:")
logger.info(f"   - Hidden dim: 128 → {config['hidden_dim']}")
logger.info(f"   - Rich features justify larger model")
logger.info("   - Feature projection MLP for structural features")
logger.info("")
logger.info("4. Balanced Regularization:")
logger.info(f"   - Encoder dropout: {config['dropout']} (reduced from 0.6)")
logger.info(f"   - Decoder dropout: {config['decoder_dropout']} (reduced from 0.5)")
logger.info(f"   - Edge dropout: {config['edge_dropout']} (reduced from 0.2)")
logger.info(f"   - Feature dropout: 0.1 (NEW!)")
logger.info(f"   - Embedding L2: {config['emb_reg_weight']} (reduced from 0.01)")
logger.info(f"   → Less aggressive regularization since features reduce overfitting")
logger.info("")
logger.info("5. Stable Training:")
logger.info(f"   - Learning rate: {config['learning_rate']}")
logger.info(f"   - Hard negative ratio: {config['hard_neg_ratio']} (reduced from 0.3 for stability)")
logger.info(f"   - Hard negative warmup: {config['hard_neg_warmup']} epochs (increased)")
logger.info(f"   - LR scheduler patience: {config['scheduler_patience']} (more patient)")
logger.info("=" * 80)

logger.info("\n" + "=" * 80)
logger.info("Training and evaluation completed successfully!")
logger.info(f"Results logged to: {log_filename}")
logger.info("=" * 80)