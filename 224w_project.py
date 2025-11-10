import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, TransformerConv
from torch_geometric.utils import negative_sampling
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
test_pos  = split_edge['test']['edge'].to(device)

num_nodes = data.num_nodes

# Construct graph using *only* training edges (IMPORTANT: prevents leakage)
data.edge_index = train_pos.t().contiguous().to(device)

evaluator = Evaluator(name='ogbl-ddi')

class BaseModel(nn.Module):
    def decode(self, z, edge):
        src, dst = edge[:, 0], edge[:, 1]
        return (z[src] * z[dst]).sum(dim=1)

class GCN(BaseModel):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def encode(self, edge_index):
        # Random non-trainable node features → prevents memorization
        x = torch.randn(num_nodes, self.hidden_dim, device=device)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(BaseModel):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def encode(self, edge_index):
        x = torch.randn(num_nodes, self.hidden_dim, device=device)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GraphTransformer(BaseModel):
    def __init__(self, hidden_dim, heads=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads)

    def encode(self, edge_index):
        x = torch.randn(num_nodes, self.hidden_dim, device=device)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def get_loss(model, edge_index, pos_edges):
    z = model.encode(edge_index)
    pos_score = model.decode(z, pos_edges)

    neg_edges = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=pos_edges.size(0)
    ).to(device)

    neg_score = model.decode(z, neg_edges)

    # Margin ranking hinge loss
    loss = -torch.log(torch.sigmoid(pos_score)).mean() - torch.log(1 - torch.sigmoid(neg_score)).mean()
    return loss

def evaluate(model):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.edge_index)

        # Positive scores
        pos_scores = model.decode(z, test_pos).view(-1).cpu()

        # Negative sampling + batch to avoid OOM
        neg_test = negative_sampling(
            edge_index=data.edge_index.cpu(),
            num_nodes=num_nodes,
            num_neg_samples=test_pos.size(0)
        )
        neg_test = neg_test.t().to(device)

        neg_scores_list = []
        batch = 200000  # safe chunk size
        for i in range(0, neg_test.size(0), batch):
            chunk = neg_test[i:i+batch]
            neg_scores_list.append(model.decode(z, chunk).view(-1).cpu())

        neg_scores = torch.cat(neg_scores_list)

        result = evaluator.eval({
            'y_pred_pos': pos_scores,
            'y_pred_neg': neg_scores,
        })

        return result['hits@20']

def train_model(name, model, epochs=1000, lr=0.001):
    logger.info(f"Starting training for {name} (epochs={epochs}, lr={lr})")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(data.edge_index)
        neg_samples = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=num_nodes,
            num_neg_samples=train_pos.size(0)
        ).t().to(device)

        pos_out = model.decode(z, train_pos).sigmoid()
        neg_out = model.decode(z, neg_samples).sigmoid()

        loss = -torch.log(pos_out + 1e-15).mean() - torch.log(1 - neg_out + 1e-15).mean()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            logger.info(f"{name} Epoch {epoch}/{epochs} - Loss = {loss.item():.4f}")

    hits20 = evaluate(model)
    logger.info(f"{name}: Hits@20 = {hits20:.4f}")
    return hits20

gcn_hits  = train_model("GCN", GCN(128))
sage_hits = train_model("GraphSAGE", GraphSAGE(128))
gt_hits   = train_model("GraphTransformer", GraphTransformer(128))

logger.info("\n==== FINAL RESULTS ====")
logger.info(f"GCN Hits@20       = {gcn_hits:.4f}")
logger.info(f"GraphSAGE Hits@20 = {sage_hits:.4f}")
logger.info(f"Transformer Hits@20 = {gt_hits:.4f}")

@torch.no_grad()
def debug_evaluator(model):
    model = model.to(device)                # ✅ move model to cuda
    model.eval()

    z = model.encode(data.edge_index)       # now both live on GPU

    pos_scores = model.decode(z, test_pos).view(-1).cpu()

    neg_test = negative_sampling(
        edge_index=data.edge_index.cpu(),   # ✅ negative_sampling works on CPU
        num_nodes=num_nodes,
        num_neg_samples=test_pos.size(0),
    )
    neg_test = neg_test.to(device)          # ✅ then move back to CUDA
    neg_scores = model.decode(z, neg_test).view(-1).cpu()

    result = evaluator.eval({
        'y_pred_pos': pos_scores,
        'y_pred_neg': neg_scores,
    })

    logger.info(f"Returned evaluator keys: {result.keys()}")
    return result

debug_evaluator(GCN(32))
logger.info("Training and evaluation completed successfully!")