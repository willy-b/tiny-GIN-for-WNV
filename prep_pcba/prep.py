import numpy as np
from ogb.io.save_dataset import DatasetSaver
from ogb.utils.mol import smiles2graph
import pandas as pd
import os

# only need to run this once when setting up your workspace
# defaults to original dataset aid 577 if not specified but upstream caller could also pass 588689
def convert_aid_into_ogb_dataset(use_scaffold_split=False, scaffold_split_seed=0, aid_id=577):
   if int(aid_id) != 577 and int(aid_id) != 588689:
      raise Exception("unsupported AID ID; must be 577 or 588689 at present")
   aid_data = pd.read_csv(f"prep_pcba/AID_{aid_id}_datatable.csv")
   smiles_entry_tags = aid_data["PUBCHEM_RESULT_TAG"]
   if int(aid_id) == 577:
      assert int(smiles_entry_tags[3]) == 1 # 1-based index of data starts at index 3
      assert int(smiles_entry_tags[4]) == 2 # 1-based index of data starts at index 3
      assert int(smiles_entry_tags[5]) == 3 # 1-based index of data starts at index 3
   # Instead of just skipping the first 3 post-header lines
   # (which have empty columns of interest for us so were already skipped) in processing loop,
   # we can skip them intentionally to be stricter on errors later (not allow any).
   # We know our input, it is a file in the repo under our control that has been human reviewed,
   # so we aren't expecting anything too surprising here.
   FIXED_HEADER_LINES_OFFSET = 4 if int(aid_id) == 588689 else 3
   ds = DatasetSaver(f"ogbg-pcba-aid-{aid_id}", is_hetero=False, version=0, root="local")
   graphs = []
   labels = []
   smiles_used = []
   assert len(aid_data["PUBCHEM_EXT_DATASOURCE_SMILES"]) == len(aid_data["PUBCHEM_ACTIVITY_OUTCOME"])
   for idx in range(FIXED_HEADER_LINES_OFFSET, len(aid_data["PUBCHEM_EXT_DATASOURCE_SMILES"])):
      smile = aid_data["PUBCHEM_EXT_DATASOURCE_SMILES"][idx]
      label = aid_data["PUBCHEM_ACTIVITY_OUTCOME"][idx]
      if type(smile) != type("string") or len(smile) == 0:
        raise Exception(f"empty SMILE in unexpected location, line {idx}")
      if type(label) != type("string") or len(label) == 0 or (label != 'Active' and label != 'Inactive' and label != 'Inconclusive'):
        raise Exception(f"Found unreadable '{label}' for smile '{smile}', skipping. If smile is nonempty then dataset may be corrupted.")
      graph = smiles2graph(smile)
      smiles_used.append(smile)
      graphs.append(graph)
      labels.append(label)
   ds.save_graph_list(graphs)
   labels_numeric = np.array([1 if label == "Active" else 0 for label in labels])
   labels_numeric = labels_numeric.reshape(-1, 1) # labels_numeric.shape == (65239, 1)
   ds.save_target_labels(labels_numeric)
   adequate_split = False
   num_total_active = labels_numeric.sum()
   # we have very low counts so let's not have any less than the expected active molecule count
   expected_active_in_test_and_validation = round(0.1*num_total_active)
   if use_scaffold_split:
      # per MoleculeNet paper https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a
      # "Scaffold splitting splits the samples based on their two-dimensional structural frameworks,[62] as implemented in RDKit.[63] Since scaffold splitting attempts to separate structurally different molecules into different subsets, it offers a greater challenge for learning algorithms than the random split."
      # They contributed their ScaffoldSplitter to the DeepChem library, so we use that implementation (which wraps RDKit's MurckoScaffold)
      # "MoleculeNet contributes the code for these splitting methods into DeepChem. Users of the library can use these splits on new datasets with short library calls."
      # https://deepchem.readthedocs.io/en/2.8.0/api_reference/splitters.html#scaffoldsplitter
      import deepchem as dc
      Xs = np.zeros(len(graphs)) # we do not need to add our molecule representation as we just use this method to get the split indices and keep data saved in OGB format
      # here just looking at split by this tool
      Ys = labels_numeric
      frac_train = 0.80
      frac_valid = 0.10
      frac_test = 0.10
      scaffoldsplitter = dc.splits.ScaffoldSplitter()
      while not adequate_split:
         # In https://github.com/deepchem/deepchem/blob/d5b293934d427062f52e2d92c1569d53d10418f9/deepchem/splits/splitters.py#L1541 
         # the DeepChem team silently ignores their "seed" argument to "split", so we need to permute first
         np.random.seed(scaffold_split_seed)
         random_perm_idx = np.random.permutation([idx for idx in range(len(graphs))])
         permuted_smiles = [smiles_used[idx] for idx in random_perm_idx]
         permuted_Ys = [Ys[idx] for idx in random_perm_idx]
         # next line's variable name is out of date, scoped to these two consecutive lines, not specific to any particular AID number (could be pcba_aid_deepchem_dataset_without_features)
         pcba_aid_577_deepchem_dataset_without_features = dc.data.DiskDataset.from_numpy(X=Xs,y=permuted_Ys,w=np.zeros(len(graphs)),ids=permuted_smiles)
         train_split, val_split, test_split = scaffoldsplitter.split(pcba_aid_577_deepchem_dataset_without_features, frac_train, frac_valid, frac_test, seed=scaffold_split_seed)
         train_split = random_perm_idx[train_split]
         val_split = random_perm_idx[val_split]
         test_split = random_perm_idx[test_split]
         split_dict = {"train": np.array(train_split), "valid": np.array(val_split), "test": np.array(test_split)}
         num_active_train = labels_numeric[train_split].sum()
         num_active_test = labels_numeric[test_split].sum()
         num_active_val = labels_numeric[val_split].sum()
         adequate_split = num_active_test >= expected_active_in_test_and_validation and num_active_val >= expected_active_in_test_and_validation         
         if not adequate_split:
            print(f"resplitting as only had {num_active_test} active molecules in the test set and {num_active_val} active molecules in validation set (with {num_active_train} training set actives), incrementing scaffold_split_seed from {scaffold_split_seed} to {scaffold_split_seed+1}")
            scaffold_split_seed += 1
         else:
            print(f"accepting the split: {num_active_test} active molecules in the test set and {num_active_val} active molecules in validation set (with {num_active_train} training set actives)")
      ds.save_split(split_dict, "scaffold-80-10-10")
   else:
      while not adequate_split:
         random_perm_idx = np.random.permutation([idx for idx in range(len(graphs))])
         end_of_train_split = int(len(random_perm_idx)*0.80)
         start_of_val_split = end_of_train_split+1
         end_of_val_split = int(len(random_perm_idx)*0.90)
         start_of_test_split = end_of_val_split + 1
         train_split = np.array(random_perm_idx[:start_of_val_split])
         val_split = np.array(random_perm_idx[start_of_val_split:start_of_test_split])
         test_split = np.array(random_perm_idx[start_of_test_split:])
         split_dict = {"train": train_split, "valid": val_split, "test": test_split}
         num_active_train = labels_numeric[train_split].sum()
         num_active_test = labels_numeric[test_split].sum()
         num_active_val = labels_numeric[val_split].sum()
         adequate_split = num_active_test >= expected_active_in_test_and_validation and num_active_val >= expected_active_in_test_and_validation
         if not adequate_split:
            print(f"resplitting as only had {num_active_test} active molecules in the test set and {num_active_val} active molecules in validation set (with {num_active_train} training set actives)")
         else:
            print(f"accepting the split: {num_active_test} active molecules in the test set and {num_active_val} active molecules in validation set (with {num_active_train} training set actives)")
      ds.save_split(split_dict, "random-80-10-10")
   # regardless of split, set same task info
   ds.save_task_info("classification", "rocauc", num_classes=2)

   # recall we are creating a local unofficial dataset so we can test PCBA AID 577 or PCBA AID 588689 with code written to work with OGB formatted datasets but we have not actually submitted either of these new datasets into OGB, so no mapping dir is required, however the OGB code requires this step is run
   os.mkdir("dummy_mapping_dir")
   with open("dummy_mapping_dir/README.md", "w") as f:
      f.write("The mapping here is not used: This is a local unofficial dataset so we can test PCBA AID 577 or PCBA AID 588689 with code written to work with OGB formatted datasets but we have not submitted this PCBA AID 577 or PCBA AID 588689 derived dataset for consideration with OGB yet at this time, so no mapping dir is required, however the OGB code requires that this mapping folder and this README.md exist.")
      f.close()
   ds.copy_mapping_dir("dummy_mapping_dir")
   os.remove("dummy_mapping_dir/README.md")
   os.rmdir("dummy_mapping_dir")

   meta_dict = ds.get_meta_dict()
   print(f"meta_dict: {meta_dict}")
   # example for PCBA AID 577 would be:
   #{'version': 0, 'dir_path': 'ogbg_pcba_aid_577_ogbg_pcba_aid_577/pcba_aid_577', 'binary': 'True', 'num tasks': 1, 'num classes': 2, 'task type': 'classification', 'eval metric': 'rocauc', 'add_inverse_edge': 'False', 'split': 'random-80-10-10', 'download_name': 'pcba_aid_577', 'url': 'https://snap.stanford.edu/ogb/data/graphproppred/pcba_aid_577.zip', 'has_node_attr': 'True', 'has_edge_attr': 'True', 'additional node files': 'None', 'additional edge files': 'None', 'is hetero': 'False'}
   return meta_dict
