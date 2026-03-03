import pickle
import argparse
import torch
from utils.data_loader import FileLoader
import pandas as pd

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-data_dir', required=True,
                        help='Path to train dataset directory (e.g. dataset/PDBBind_v2016_refined_set)')
    parser.add_argument('-data_index', required=True,
                        help='Path to train index file (e.g. dataset/INDEX_refined_data.2016)')
    parser.add_argument('-data_val_dir', required=True,
                        help='Path to val/test dataset directory (e.g. dataset/coreset)')
    parser.add_argument('-data_val_index', required=True,
                        help='Path to val/test index file (e.g. dataset/CoreSet.dat)')
    parser.add_argument('-output', default='dataset/preprocessed.pkl',
                        help='Output path for the preprocessed pickle file')
    parser.add_argument('-out_csv', default='dataset/prep_times.csv',
                        help='Output path for the preprocessing time log csv file')
    parser.add_argument('-atom_keys', default='utils/PDB_Atom_Keys.csv',
                        help='Path to PDB_Atom_Keys.csv')
    args, _ = parser.parse_known_args()
    return args


args = get_args()

data, time_list = FileLoader(args).load_data()
with open(args.output, 'wb') as save_file:
    pickle.dump(data, save_file)

time_df = pd.DataFrame(time_list)
time_df.to_csv(args.out_csv, index=False)