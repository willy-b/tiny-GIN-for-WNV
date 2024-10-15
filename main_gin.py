from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from prep_pcba_577.prep import convert_aid_577_into_ogb_dataset

import torch
import torch_geometric
from torch_geometric.nn.norm import InstanceNorm, GraphNorm
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import MLP

from sklearn.metrics import RocCurveDisplay, roc_auc_score, PrecisionRecallDisplay, average_precision_score
import matplotlib.pyplot as plt

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
argparser.add_argument("--random_seed_for_data_splits", type=int, default=None) # default uses 0 but resplits if set explicitly
#argparser.add_argument("--hide_test_metric", action="store_true") # always hidden as still doing hyperparameter search at this stage
# Optionally, allow reweighting of loss to account for class imbalance.
# Default of 1.0 has no effect,
# setting to 1/active_molecule_prevalence would encourage recall of active molecules same as inactive at cost of precision.
# When judging by ROCAUC and/or AP using the raw y prediction scores
# to rank the results, this is not necessary but may of interest
# especially if using output as a classifier on a single example
# or online rather than ranking a list.
argparser.add_argument("--active_class_weight", type=float, default=1.0)
argparser.add_argument("--disable_graph_norm", action="store_true")
argparser.add_argument("--hold_out_addl_data_from_train_set_as_dev_for_addl_generalization_check", action="store_true")
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

set_seeds(args.random_seed_for_data_splits if args.random_seed_for_data_splits != None else 0)

# check if splits already exist
data_path = Path("local_ogbg_pcba_aid_577")
if not data_path.exists() or not data_path.is_dir():
    meta_dict = convert_aid_577_into_ogb_dataset()
else:
    if args.random_seed_for_data_splits != None:
        raise Exception(f"Data is already split into train/valid/test but `--random_seed_for_data_splits` argument is set, if you intend to split the data according to the specified seed and it is not the current split, please remove `{data_path}` folder and run again or if you want to reuse the existing split remove the `--random_seed_for_data_splits` argument. If you do not know what to do and you are seeing this message, you should probably remove the `{data_path}` folder (this is the safest route if copy pasting a command which includes that argument).")
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

split_idx = dataset.get_idx_split()
with open("train_valid_test_split_idxs_dict.pkl", "wb") as f:
    pickle.dump(split_idx, f)
print("dumped train/valid/test split indices dict to train_valid_test_split_idxs_dict.pkl (note these will be randomized each run by default so you should save this file for reproducibility)")

# note they are preshuffled and presplit into training/validation/test splits
if args.hold_out_addl_data_from_train_set_as_dev_for_addl_generalization_check and (len(split_idx["train"]) < 2 * len(split_idx["valid"])):
    raise Exception("Cannot use --hold_out_addl_data_from_train_set_as_dev_for_addl_generalization_check when validation set is greater than half the train set size due to insufficient data remaining.")

# based on args.hold_out_addl_data_from_train_set_as_dev_for_addl_generalization_check we can hold out validation split sized chunk of our train split as an extra dev set for final generalization check (not peeking at test) after doing early stopping on training using validation performance
adequate_split = False
full_train_split = split_idx["train"]
while not adequate_split:
    train_split = full_train_split if not args.hold_out_addl_data_from_train_set_as_dev_for_addl_generalization_check else full_train_split[len(split_idx["valid"]):]
    gen_labels = None if not args.hold_out_addl_data_from_train_set_as_dev_for_addl_generalization_check else np.array([d.y for d in dataset[full_train_split[:len(split_idx["valid"])]]])
    valid_labels = np.array([d.y for d in dataset[split_idx["valid"]]])
    adequate_split = (not args.hold_out_addl_data_from_train_set_as_dev_for_addl_generalization_check) or (gen_labels.sum() >= valid_labels.sum())
    if not adequate_split:
        print(f"Resplitting generalization check hold out from training data; only had {gen_labels.sum()} active molecules vs valid which has {valid_labels.sum()} active molecules")
        # we can permute this since training data is shuffled anyway
        # then we resample a valid split sized subset if hold_out_addl_data_from_train_set_as_dev_for_addl_generalization_check is set until we get enough active molecules in our sample to score the result
        full_train_split = np.random.permutation(full_train_split)
    else:
        if args.hold_out_addl_data_from_train_set_as_dev_for_addl_generalization_check:
            print(f"Found working generalization check hold out from training data; has {gen_labels.sum()} active molecules vs valid which has {valid_labels.sum()} active molecules")

valid_split = split_idx["valid"]

dev_split = None if not args.hold_out_addl_data_from_train_set_as_dev_for_addl_generalization_check else full_train_split[:len(split_idx["valid"])]# NOT TEST, this is taking some of the existing training split
train_loader = DataLoader(dataset[train_split], batch_size=config["batch_size"], shuffle=True)
valid_loader = DataLoader(dataset[valid_split], batch_size=config["batch_size"], shuffle=False)
dev_loader = None
if args.hold_out_addl_data_from_train_set_as_dev_for_addl_generalization_check:
    print(f"Held out {len(dev_split)} datapoints from training split as a final generalization check after selecting best model (similar to early stopping) using valid split")
    dev_loader = DataLoader(dataset[dev_split], batch_size=config["batch_size"], shuffle=False)

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

def eval(model, device, loader, evaluator, save_model_results=False, save_filename=None, split_indices=[], plot_metrics=False, figure_save_tag=""):
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
  if plot_metrics:
      # an error here about `plot_chance_level` likely indicates scikit-learn dependency is not >=1.3
      RocCurveDisplay.from_predictions(y_true, y_pred, plot_chance_level=True)
      using_graphnorm_filename_string = "_and_GraphNorm" if config['use_graph_norm'] else ""
      using_graphnorm_title_string = " with GraphNorm" if config['use_graph_norm'] else ""
      plt.title(f"Predicting if molecules inhibit West Nile Virus NS2bNS3 Proteinase\n{sum(p.numel() for p in best_model.parameters())} parameter {config['num_layers']}-hop GIN{using_graphnorm_title_string} and hidden dimension {config['hidden_dim']}")
      if figure_save_tag != "":
          figure_save_tag = f"_{figure_save_tag}"
      rocauc_figure_filename = f"WNV_NS2bNS3_Proteinase_Inhibition_Prediction_using_{config['num_layers']}-hop_GIN_hidden_dim_{config['hidden_dim']}{using_graphnorm_filename_string}_ROC_CURVE{figure_save_tag}.png"
      plt.savefig(rocauc_figure_filename)
      rocauc_score = roc_auc_score(y_true, y_pred)
      print(f"Generated {rocauc_figure_filename}\nshowing Receiver Operating Characteristic Area Under the Curve (ROCAUC) score of {rocauc_score:.6f} (chance level is {0.5:.6f})")
      plt.show()
      PrecisionRecallDisplay.from_predictions(y_true, y_pred, plot_chance_level=True)
      plt.title(f"Predicting if molecules inhibit West Nile Virus NS2bNS3 Proteinase\n{sum(p.numel() for p in best_model.parameters())} parameter {config['num_layers']}-hop GIN{using_graphnorm_title_string} and hidden dimension {config['hidden_dim']}")
      precision_recall_display_filename = f"WNV_NS2bNS3_Proteinase_Inhibition_Prediction_using_{config['num_layers']}-hop_GIN_hidden_dim_{config['hidden_dim']}{using_graphnorm_filename_string}_PRC_CURVE{figure_save_tag}.png"
      plt.savefig(precision_recall_display_filename)
      ap_score = average_precision_score(y_true, y_pred)
      chance_ap = y_true.sum()/len(y_true)
      print(f"Generated {precision_recall_display_filename}\nshowing average precision (AP) score of {ap_score:.6f} (chance level is {chance_ap:.6f})")
      plt.show()
  return evaluator.eval(input_dict)

model = GINGraphPropertyModel(config['hidden_dim'], dataset.num_tasks, config['num_layers'], config['dropout']).to(device)
print(f"parameter count: {sum(p.numel() for p in model.parameters())}")
model.reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.active_class_weight]))
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
valid_metric = eval(best_model, device, valid_loader, evaluator, save_model_results=True, save_filename=f"gin_{config['dataset_id']}_valid", split_indices=valid_split, plot_metrics=True, figure_save_tag="valid")[dataset.eval_metric]
if args.hold_out_addl_data_from_train_set_as_dev_for_addl_generalization_check:
    dev_metric = eval(best_model, device, dev_loader, evaluator, save_model_results=True, save_filename=f"gin_{config['dataset_id']}_train_split_holdout_as_dev", split_indices=dev_split, plot_metrics=True, figure_save_tag="train_subset_holdout_as_dev")[dataset.eval_metric]

#test_metric  = eval(best_model, device, test_loader, evaluator, save_model_results=True, save_filename=f"gin_{config['dataset_id']}_test", split_indices=split_idx["test"])[dataset.eval_metric]

print(f'Best model for {config["dataset_id"]} (eval metric {dataset.eval_metric}): '
      f'Train: {train_metric:.6f}, '
      f'Valid: {valid_metric:.6f} ')
      #f'Test: {test_metric:.6f}')
if args.hold_out_addl_data_from_train_set_as_dev_for_addl_generalization_check:
    print(f"Train subset held out as dev for generalization check: {dev_metric:.6f}")
print(f"parameter count: {sum(p.numel() for p in best_model.parameters())}")

