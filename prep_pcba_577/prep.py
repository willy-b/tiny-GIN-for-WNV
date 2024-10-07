import numpy as np
from ogb.io.save_dataset import DatasetSaver
from ogb.utils.mol import smiles2graph
import pandas as pd
import os

# only need to run this once when setting up your workspace
def convert_aid_577_into_ogb_dataset():
   aid577 = pd.read_csv("prep_pcba_577/AID_577_datatable.csv")
   ds = DatasetSaver("ogbg-pcba-aid-577", is_hetero=False, version=0, root="local")
   graphs = []
   for idx in range(len(aid577["PUBCHEM_EXT_DATASOURCE_SMILES"])):
      smile = aid577["PUBCHEM_EXT_DATASOURCE_SMILES"][idx]
      if type(smile) != type("string") or len(smile) == 0:
         continue
      graph = smiles2graph(smile)
      graphs.append(graph)
   ds.save_graph_list(graphs)
   labels = []
   for idx in range(len(aid577["PUBCHEM_ACTIVITY_OUTCOME"])):
      label = aid577["PUBCHEM_ACTIVITY_OUTCOME"][idx]
      if type(label) != type("string") or len(label) == 0:
        smile = aid577["PUBCHEM_EXT_DATASOURCE_SMILES"][idx]
        print(f"No label for smile {smile}. If smile is nonempty, please stop and check that the dataset has not become corrupted!")
        if type(smile) == type("string") and len(smile) == 0:
            raise Error("No label for nonempty smile string!")
        continue
      labels.append(label)
   labels_numeric = np.array([0 if label != "Active" else 1 for label in labels])
   labels_numeric = labels_numeric.reshape(-1, 1) # labels_numeric.shape == (65239, 1)
   ds.save_target_labels(labels_numeric)
   adequate_split = False
   while not adequate_split:
      num_total_active = labels_numeric.sum()
      # we have very low counts so let's not have any less than the expected active molecule count
      expected_active_in_test_and_validation = round(0.1*num_total_active)
      random_perm_idx = np.random.permutation([idx for idx in range(len(graphs))])
      end_of_train_split = int(len(random_perm_idx)*0.80)
      start_of_val_split = end_of_train_split+1
      end_of_val_split = int(len(random_perm_idx)*0.90)
      start_of_test_split = end_of_val_split + 1
      train_split = np.array(random_perm_idx[:start_of_val_split])
      val_split = np.array(random_perm_idx[start_of_val_split:start_of_test_split])
      test_split = np.array(random_perm_idx[start_of_test_split:])
      split_dict = {"train": train_split, "valid": val_split, "test": test_split}
      num_active_test = labels_numeric[test_split].sum()
      num_active_val = labels_numeric[val_split].sum()
      adequate_split = num_active_test >= expected_active_in_test_and_validation and num_active_val >= expected_active_in_test_and_validation
      if not adequate_split:
         print(f"resplitting as only had {num_active_test} active molecules in the test set and {num_active_val} active molecules in validation set")
      else:
         print(f"accepting the split: {num_active_test} active molecules in the test set and {num_active_val} active molecules in validation set")
   ds.save_split(split_dict, "random-80-10-10")
   ds.save_task_info("classification", "rocauc", num_classes=2)

   # recall we are creating a local unofficial dataset so we can test PCBA AID 577 with code written to work with OGB formatted datasets but I don't actually intend to put PCBA AID 577 into OGB, so no mapping dir is required, however the OGB code requires this step is run
   os.mkdir("dummy_mapping_dir")
   with open("dummy_mapping_dir/README.md", "w") as f:
      f.write("The mapping here is not used: This is a local unofficial dataset so we can test PCBA AID 577 with code written to work with OGB formatted datasets but we do not actually intend to submit this PCBA AID 577 dataset for consideration with OGB at this time, so no mapping dir is required, however the OGB code requires that this mapping folder and this README.md exist.")
      f.close()
   ds.copy_mapping_dir("dummy_mapping_dir")
   os.remove("dummy_mapping_dir/README.md")
   os.rmdir("dummy_mapping_dir")

   meta_dict = ds.get_meta_dict()
   print(f"meta_dict: {meta_dict}")
   #{'version': 0, 'dir_path': 'ogbg_pcba_aid_577_ogbg_pcba_aid_577/pcba_aid_577', 'binary': 'True', 'num tasks': 1, 'num classes': 2, 'task type': 'classification', 'eval metric': 'rocauc', 'add_inverse_edge': 'False', 'split': 'random-80-10-10', 'download_name': 'pcba_aid_577', 'url': 'https://snap.stanford.edu/ogb/data/graphproppred/pcba_aid_577.zip', 'has_node_attr': 'True', 'has_edge_attr': 'True', 'additional node files': 'None', 'additional edge files': 'None', 'is hetero': 'False'}
   return meta_dict

