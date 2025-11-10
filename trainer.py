import argparse
import os
import random
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch_geometric.nn import SAGEConv, GCNConv
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling


# -------------------------------
# Utils
# -------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LinkPredictor(nn.Module):
    """Simple dot-product decoder."""
    def forward(self, z: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        src, dst = edge[:, 0], edge[:, 1]
        return (z[src] * z[dst]).sum(dim=1)


class SAGEModel(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3,
                 use_bn: bool = True, model_type: str = "sage"):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        self.dropout = dropout
        self.use_bn = use_bn
        self.model_type = model_type

        convs = []
        norms = []
        for _ in range(num_layers):
            if model_type == "sage":
                convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif model_type == "gcn":
                convs.append(GCNConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            norms.append(nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity())
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.predictor = LinkPredictor()

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.emb.weight
        for conv, norm in zip(self.convs, self.norms):
            if isinstance(conv, GCNConv):
                x_new = conv(x, edge_index)
            else:
                x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            # Residual
            if x_new.shape == x.shape:
                x = x + x_new
            else:
                x = x_new
        return x

    def decode(self, z: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        return self.predictor(z, edge)


@torch.no_grad()
def evaluate_hits20(model: nn.Module, edge_index: torch.Tensor, z: torch.Tensor,
                    pos_edge: torch.Tensor, neg_edge: torch.Tensor, evaluator: Evaluator) -> float:
    model.eval()
    pos_scores = model.decode(z, pos_edge).view(-1).cpu()
    neg_scores = model.decode(z, neg_edge).view(-1).cpu()
    result = evaluator.eval({
        'y_pred_pos': pos_scores,
        'y_pred_neg': neg_scores,
    })
    return float(result['hits@20'])


def train_one_seed(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = PygLinkPropPredDataset('ogbl-ddi')
    data = dataset[0]
    split_edge = dataset.get_edge_split()

    # Use only training edges for message passing to avoid leakage
    train_pos = split_edge['train']['edge'].to(device)
    valid_pos = split_edge['valid']['edge'].to(device)
    test_pos = split_edge['test']['edge'].to(device)

    valid_neg = split_edge['valid']['edge_neg'].to(device)
    test_neg = split_edge['test']['edge_neg'].to(device)

    num_nodes = data.num_nodes
    edge_index = train_pos.t().contiguous().to(device)

    model = SAGEModel(
        num_nodes=num_nodes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_bn=not args.no_bn,
        model_type=args.model,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    evaluator = Evaluator(name='ogbl-ddi')

    best_val = -1.0
    best_test = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(edge_index)

        # Sample negatives equal to number of positives per epoch
        neg = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=train_pos.size(0),
            method='sparse'
        ).t().to(device)

        pos_out = model.decode(z, train_pos)
        neg_out = model.decode(z, neg)

        # BCE with logits, balanced positives/negatives
        pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        if epoch % args.eval_steps == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                z_eval = model.encode(edge_index)
                val_hits = evaluate_hits20(model, edge_index, z_eval, valid_pos, valid_neg, evaluator)
                test_hits = evaluate_hits20(model, edge_index, z_eval, test_pos, test_neg, evaluator)

            improved = val_hits > best_val
            if improved:
                best_val = val_hits
                best_test = test_hits
                best_epoch = epoch
                epochs_no_improve = 0
                # Save checkpoint
                if args.ckpt_dir:
                    os.makedirs(args.ckpt_dir, exist_ok=True)
                    torch.save({
                        'model_state': model.state_dict(),
                        'epoch': epoch,
                        'val_hits20': val_hits,
                        'test_hits20': test_hits,
                        'args': vars(args),
                    }, os.path.join(args.ckpt_dir, f"best_{args.model}.pt"))
            else:
                epochs_no_improve += 1

            print(f"Epoch {epoch:04d} | loss {loss.item():.4f} | val@20 {val_hits:.4f} | test@20 {test_hits:.4f} | best val {best_val:.4f} (ep {best_epoch})")

            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {epochs_no_improve} evals)")
                break

    return {
        'best_val_hits20': best_val,
        'best_test_hits20': best_test,
        'best_epoch': float(best_epoch),
    }


def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI Link Prediction Trainer')
    parser.add_argument('--model', type=str, default='sage', choices=['sage', 'gcn'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--no_bn', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_val = []
    all_test = []
    for seed in args.seeds:
        print("\n" + "=" * 30)
        print(f"Running seed {seed}")
        print("=" * 30)
        set_seed(seed)
        stats = train_one_seed(args)
        all_val.append(stats['best_val_hits20'])
        all_test.append(stats['best_test_hits20'])
        print(f"Seed {seed} -> best val@20 {stats['best_val_hits20']:.4f} | best test@20 {stats['best_test_hits20']:.4f} (epoch {int(stats['best_epoch'])})")

    val_mean = float(np.mean(all_val))
    val_std = float(np.std(all_val))
    test_mean = float(np.mean(all_test))
    test_std = float(np.std(all_test))

    print("\n==== FINAL (mean ± std) across seeds ====")
    print(f"Val Hits@20:  {val_mean:.4f} ± {val_std:.4f}")
    print(f"Test Hits@20: {test_mean:.4f} ± {test_std:.4f}")


if __name__ == '__main__':
    main()
