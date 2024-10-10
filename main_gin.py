from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from prep_pcba_577.prep import convert_aid_577_into_ogb_dataset

import torch
import torch_geometric
from torch_geometric.nn.norm import InstanceNorm, GraphNorm
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import MLP

import pandas as pd
import copy
import numpy as np
import random
from tqdm import tqdm
import pickle
from pathlib import Path

import argparse

argparser = argparse.ArgumentParser()
# Set this to 'cpu' if you NEED to reproduce exact numbers.
argparser.add_argument("--device", type=str, default='cpu')
argparser.add_argument("--num_layers", type=int, default=2)
argparser.add_argument("--hidden_dim", type=int, default=56)
argparser.add_argument("--learning_rate", type=float, default=0.001)
argparser.add_argument("--dropout_p", type=float, default=0.5)
argparser.add_argument("--epochs", type=int, default=120)
argparser.add_argument("--batch_size", type=int, default=1024) # needs large batches since < 0.2% of the data is active molecules otherwise almost all batches will be all inactives
argparser.add_argument("--weight_decay", type=float, default=5e-4) # learning from very few examples (<100 active examples total), increase regularization to avoid overfitting
# note that the random seed for your first execution determines your split
# so if you want to reproduce official results, clear out the local_ogbg_pcba_aid_577 folder (generated on first run)
# and run the seed 0 first!
# alternatively, if you want to try different data splits, delete the folder and use a different seed.
argparser.add_argument("--random_seed", type=int, default=None)
#argparser.add_argument("--hide_test_metric", action="store_true") # always hidden as still doing hyperparameter search at this stage
argparser.add_argument("--disable_graph_norm", action="store_true")
args = argparser.parse_args()

# Let's set a random seed for reproducibility
# If using a GPU choosing the same seed cannot be used to guarantee
# that one gets the same result from run to run,
# but may still be useful to ensure one is starting with different seeds.
def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if args.random_seed != None:
    set_seeds(args.random_seed)

# check if splits already exist
data_path = Path("local_ogbg_pcba_aid_577")
if not data_path.exists() or not data_path.is_dir():
    meta_dict = convert_aid_577_into_ogb_dataset()
else:
    # default data is available, use that
    meta_dict = {'version': 0, 'dir_path': 'local_ogbg_pcba_aid_577/pcba_aid_577', 'binary': 'True', 'num tasks': 1, 'num classes': 2, 'task type': 'classification', 'eval metric': 'rocauc', 'add_inverse_edge': 'False', 'split': 'random-80-10-10', 'download_name': 'pcba_aid_577', 'url': 'https://snap.stanford.edu/ogb/data/graphproppred/pcba_aid_577.zip', 'has_node_attr': 'True', 'has_edge_attr': 'True', 'additional node files': 'None', 'additional edge files': 'None', 'is hetero': 'False'}
dataset = PygGraphPropPredDataset(name="ogbg-pcba-aid-577", root="local", transform=None, meta_dict=meta_dict)
evaluator = Evaluator(name="ogbg-molhiv") # intentionally use ogbg-molhiv evaluator for ogbg-pcba-aid-577 since we have put data in same format (single task, binary output molecular/graph property prediction)

# for looking up SMILES strings to include in the output CSV with scores
aid577 = pd.read_csv("prep_pcba_577/AID_577_datatable.csv")
smiles_list = aid577["PUBCHEM_EXT_DATASOURCE_SMILES"].values[3:] # see below, 1-based index of this data starts at index 3, so that should be zero-index in our zero-based index smiles list
smiles_entry_tags = aid577["PUBCHEM_RESULT_TAG"]
assert int(smiles_entry_tags[3]) == 1 # 1-based index of data starts at index 3
assert int(smiles_entry_tags[4]) == 2 # 1-based index of data starts at index 3
assert int(smiles_entry_tags[5]) == 3 # 1-based index of data starts at index 3

if args.random_seed != None:
    set_seeds(args.random_seed)

config = {
 # Set this to 'cpu' if you NEED to reproduce exact numbers.
 'device': args.device,
 'dataset_id': 'ogbg-pcba-aid-577',
 'num_layers': args.num_layers, # 2
 'hidden_dim': args.hidden_dim, # 56
 'dropout': args.dropout_p, # 0.50
 'learning_rate': args.learning_rate, # 0.001
 'epochs': args.epochs, # 120, this problem may need more time than ogbg-molhiv
 'batch_size': args.batch_size, # 1024 ; needs larger batches since < 0.2% of the data is active molecules otherwise almost all batches will be entirely inactive examples for that gradient descent step
 'weight_decay': args.weight_decay # 5e-4 ; learning from very few examples (<100 active examples total in the training data), increase regularization to avoid overfitting
}
device = config["device"]

# note these splits are randomized each time right now, unlike actual OGB datasets, may have to modify to enforce a minimum number of positives persplit given low base rate and randomness
split_idx = dataset.get_idx_split()
with open("train_valid_test_split_idxs_dict.pkl", "wb") as f:
    pickle.dump(split_idx, f)
print("dumped train/valid/test split indices dict to train_valid_test_split_idxs_dict.pkl (note these will be randomized each run by default so you should save this file for reproducibility)")
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=config["batch_size"], shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=config["batch_size"], shuffle=False)

config["use_graph_norm"] = not args.disable_graph_norm # on by default

print(f"config: {config}")

# same class as in https://github.com/willy-b/tiny-GIN-for-ogbg-molhiv/ ,
# though this supports GraphNorm,
# TODO factor out to shared dependency for future such repos
# (not updating ogbg-molhiv repo much as it is snapshot of what is on leaderboard)
# computes a node embedding using GINConv layers, then uses pooling to predict graph level properties
class GINGraphPropertyModel(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout_p, return_node_embed=False):
      super(GINGraphPropertyModel, self).__init__()
      # fields used for computing node embedding
      self.node_encoder = AtomEncoder(hidden_dim)
      self.return_node_embed = return_node_embed
      self.convs = torch.nn.ModuleList(
          [torch_geometric.nn.conv.GINConv(MLP([hidden_dim, hidden_dim, hidden_dim])) for idx in range(0, num_layers)]
      )
      if config['use_graph_norm']:
          self.bns = torch.nn.ModuleList(
              [GraphNorm(hidden_dim) for idx in range(0, num_layers - 1)],
          )
      else:
          self.bns = torch.nn.ModuleList(
              [torch.nn.BatchNorm1d(hidden_dim) for idx in range(0, num_layers - 1)]
          )
      self.dropout_p = dropout_p
      # end fields used for computing node embedding
      # fields for graph embedding
      self.pool = global_add_pool
      self.linear_hidden = torch.nn.Linear(hidden_dim, hidden_dim)
      self.linear_out = torch.nn.Linear(hidden_dim, output_dim)
      # end fields for graph embedding
    def reset_parameters(self):
      for conv in self.convs:
        conv.reset_parameters()
      for bn in self.bns:
        bn.reset_parameters()
      self.linear_hidden.reset_parameters()
      self.linear_out.reset_parameters()
    def forward(self, x, edge_index, batch):
      #x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
      # compute node embedding
      x = self.node_encoder(x)
      for idx in range(0, len(self.convs)):
        x = self.convs[idx](x, edge_index)
        if idx < len(self.convs) - 1:
          if config['use_graph_norm']:
              x = self.bns[idx](x, batch)
          else:
              x = self.bns[idx](x)
          x = torch.nn.functional.relu(x)
          x = torch.nn.functional.dropout(x, self.dropout_p, training=self.training)
      # note x is raw logits, NOT softmax'd
      # end computation of node embedding
      if self.return_node_embed == True:
        return x
      # convert node embedding to a graph level embedding using pooling
      x = self.pool(x, batch)
      x = torch.nn.functional.dropout(x, self.dropout_p, training=self.training)
      # transform the graph embedding to the output dimension
      # MLP after graph embed ensures we are not requiring the raw pooled node embeddings to be linearly separable
      x = self.linear_hidden(x)
      x = torch.nn.functional.relu(x)
      x = torch.nn.functional.dropout(x, self.dropout_p, training=self.training)
      out = self.linear_out(x)
      return out

# can be used with multiple task outputs (like for molpcba) or single task output;
# and supports using just the first output of a multi-task model if applied to a single task (for pretraining molpcba and transferring to molhiv)
def train(model, device, data_loader, optimizer, loss_fn):
  model.train()
  for step, batch in enumerate(tqdm(data_loader, desc="Training batch")):
    batch = batch.to(device)
    if batch.x.shape[0] != 1 and batch.batch[-1] != 0:
      # ignore nan targets (unlabeled) when computing training loss.
      non_nan = batch.y == batch.y
      loss = None
      optimizer.zero_grad()
      out = model(batch.x, batch.edge_index, batch.batch)
      non_nan = non_nan[:min(non_nan.shape[0], out.shape[0])]
      batch_y = batch.y[:out.shape[0], :]
      # for crudely adapting multitask models to single task data
      if batch.y.shape[1] == 1:
        out = out[:, 0]
        batch_y = batch_y[:, 0]
        non_nan = batch_y == batch_y
        loss = loss_fn(out[non_nan].reshape(-1, 1)*1., batch_y[non_nan].reshape(-1, 1)*1.)
      else:
        loss = loss_fn(out[non_nan], batch_y[non_nan])
      loss.backward()
      optimizer.step()
  return loss.item()

def eval(model, device, loader, evaluator, save_model_results=False, save_filename=None, split_indices=[]):
  model.eval()
  y_true = []
  y_pred = []
  indices = []
  seen = set()
  for step, batch in enumerate(tqdm(loader, desc="Evaluation batch")):
      batch = batch.to(device)
      with torch.no_grad():
          pred = model(batch.x, batch.edge_index, batch.batch)
          # for crudely adapting multitask models to single task data
          if batch.y.shape[1] == 1:
            pred = pred[:, 0]
          batch_y = batch.y[:min(pred.shape[0], batch.y.shape[0])]
          y_true.append(batch_y.view(pred.shape).detach().cpu())
          y_pred.append(pred.detach().cpu())
          offset = len(seen)
          for ind in batch.batch:
              adjusted_ind = ind.item() + offset
              if not adjusted_ind in seen:
                  indices.append(adjusted_ind)
                  seen.add(adjusted_ind)
  y_true = torch.cat(y_true, dim=0).numpy()
  y_pred = torch.cat(y_pred, dim=0).numpy()
  input_dict = {"y_true": y_true.reshape(-1, 1) if batch.y.shape[1] == 1 else y_true, "y_pred": y_pred.reshape(-1, 1) if batch.y.shape[1] == 1 else y_pred}
  if save_model_results:
      data = {
          'y_pred': y_pred.squeeze(),
          'y_true': y_true.squeeze()
      }
      # lookup smiles to add to CSV
      if len(split_indices) > 0:
          # we need the split indices to lookup the smiles
          original_indices = [split_indices[idx] for idx in indices]
          smiles = [smiles_list[idx] for idx in original_indices]
          data["smiles"] = smiles
      pd.DataFrame(data=data).to_csv('ogbg_graph_' + save_filename + '.csv', sep=',', index=False)
  return evaluator.eval(input_dict)

model = GINGraphPropertyModel(config['hidden_dim'], dataset.num_tasks, config['num_layers'], config['dropout']).to(device)
print(f"parameter count: {sum(p.numel() for p in model.parameters())}")
model.reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
loss_fn = torch.nn.BCEWithLogitsLoss()
best_model = None
best_valid_metric_at_save_checkpoint = 0
best_train_metric_at_save_checkpoint = 0

for epoch in range(1, 1 + config["epochs"]):
  if epoch == 10:
    # reduce learning rate at this point
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']*0.5, weight_decay=config['weight_decay'])
  loss = train(model, device, train_loader, optimizer, loss_fn)
  train_perf = eval(model, device, train_loader, evaluator)
  val_perf = eval(model, device, valid_loader, evaluator)
  #test_perf = eval(model, device, test_loader, evaluator)
  train_metric, valid_metric = train_perf[dataset.eval_metric], val_perf[dataset.eval_metric]#, test_perf[dataset.eval_metric]
  if valid_metric >= best_valid_metric_at_save_checkpoint and train_metric >= best_train_metric_at_save_checkpoint:
    print(f"New best validation score: {valid_metric} ({dataset.eval_metric}) without training score regression")
    best_valid_metric_at_save_checkpoint = valid_metric
    best_train_metric_at_save_checkpoint = train_metric
    best_model = copy.deepcopy(model)
  print(f'Dataset {config["dataset_id"]}, '
    f'Epoch: {epoch}, '
    f'Train: {train_metric:.6f} ({dataset.eval_metric}), '
    f'Valid: {valid_metric:.6f} ({dataset.eval_metric}), '
    #f'Test: {test_metric:.6f} ({dataset.eval_metric})'
   )

with open(f"best_{config['dataset_id']}_gin_model_{config['num_layers']}_layers_{config['hidden_dim']}_hidden.pkl", "wb") as f:
  pickle.dump(best_model, f)

train_metric = eval(best_model, device, train_loader, evaluator)[dataset.eval_metric]
valid_metric = eval(best_model, device, valid_loader, evaluator, save_model_results=True, save_filename=f"gin_{config['dataset_id']}_valid", split_indices=split_idx["valid"])[dataset.eval_metric]
#test_metric  = eval(best_model, device, test_loader, evaluator, save_model_results=True, save_filename=f"gin_{config['dataset_id']}_test", split_indices=split_idx["test"])[dataset.eval_metric]

print(f'Best model for {config["dataset_id"]} (eval metric {dataset.eval_metric}): '
      f'Train: {train_metric:.6f}, '
      f'Valid: {valid_metric:.6f} ')
      #f'Test: {test_metric:.6f}')
print(f"parameter count: {sum(p.numel() for p in best_model.parameters())}")

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
model = best_model
model.eval()
y_true = []
y_pred = []
for step, batch in enumerate(tqdm(valid_loader, desc="Evaluation batch")):
  batch = batch.to(device)
  if batch.x.shape[0] == 1:
    pass
  else:
    with torch.no_grad():
      pred = model(batch.x, batch.edge_index, batch.batch)
      # for crudely adapting multitask models to single task data
      if batch.y.shape[1] == 1:
        pred = pred[:, 0]
      batch_y = batch.y[:min(pred.shape[0], batch.y.shape[0])]
      y_true.append(batch_y.view(pred.shape).detach().cpu())
      y_pred.append(pred.detach().cpu())

y_true = torch.cat(y_true, dim=0).numpy()
y_pred = torch.cat(y_pred, dim=0).numpy()

# an error here about `plot_chance_level` likely indicates scikit-learn dependency is not >=1.3
RocCurveDisplay.from_predictions(y_true, y_pred, plot_chance_level=True)
using_graphnorm_filename_string = "_and_GraphNorm" if config['use_graph_norm'] else ""
using_graphnorm_title_string = " with GraphNorm" if config['use_graph_norm'] else ""
plt.title(f"Predicting if molecules inhibit West Nile Virus NS2bNS3 Proteinase\n{sum(p.numel() for p in best_model.parameters())} parameter {config['num_layers']}-hop GIN{using_graphnorm_title_string} and hidden dimension {config['hidden_dim']}")
plt.savefig(f"WNV_NS2bNS3_Proteinase_Inhibition_Prediction_using_{config['num_layers']}-hop_GIN_hidden_dim_{config['hidden_dim']}{using_graphnorm_filename_string}_ROC_CURVE.png")
plt.show()
PrecisionRecallDisplay.from_predictions(y_true, y_pred, plot_chance_level=True)
plt.title(f"Predicting if molecules inhibit West Nile Virus NS2bNS3 Proteinase\n{sum(p.numel() for p in best_model.parameters())} parameter {config['num_layers']}-hop GIN{using_graphnorm_title_string} and hidden dimension {config['hidden_dim']}")
plt.savefig(f"WNV_NS2bNS3_Proteinase_Inhibition_Prediction_using_{config['num_layers']}-hop_GIN_hidden_dim_{config['hidden_dim']}{using_graphnorm_filename_string}_PRC_CURVE.png")
plt.show()
