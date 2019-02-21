import pandas as pd
import numpy as np
import os
import pickle
import tqdm
from utils.path_utils import project_root

data_path = os.path.join(project_root(), 'data', 'raw', 'training')
data_path2 = os.path.join(project_root(), 'data', 'processed', 'training')

training_examples = []
training_files = [f for f in os.listdir(data_path) if f.endswith('.psv')]
training_files.sort()

for training_file in tqdm.tqdm(training_files):
    example = pd.read_csv(os.path.join(data_path, training_file), sep='|')
    example.to_csv(os.path.join(data_path2, training_file[:-4] + '.csv'), sep=',', index=False, header=example.columns)
