import pandas as pd
import torch
import os
from datetime import datetime as dt
from tqdm import tqdm
from convergence_entropy.CEDA import ceda_model



###########################################################################################
###### Basic set-up
###########################################################################################
print('CUDA:', torch.cuda.is_available())

start = dt.now()
PATH = '<path_to_data_on_linux>'
output_name = os.path.join(PATH,'ckpts','{}-recplot.pt')
RAW_DATA_PATH = os.path.join(PATH, 'data')
dataset = [os.path.join(RAW_DATA_PATH,f) for f in os.listdir(RAW_DATA_PATH) if not f.startswith('._')]

print(PATH, '\n\n')
level = [7, -1]



###########################################################################################
###### Process
###########################################################################################
start = dt.now()
for f in dataset:
    output_name_ = str(output_name).format(f.split('/')[-1].replace('.csv', ''))

    df = pd.read_csv(f)
    df = df.loc[
        ~df['utterance'].isna()
        & ~df['next_utterance'].isna()
    ]

    print(df['speaker'].unique(), df['next_speaker'].unique())

    meta_data_cols = list(df)

    GRAPH = ceda_model(
        sigma=1.5,
        device='cuda',
        wv_model='roberta-base',
        wv_layers=level
    )

    GRAPH.fit(df['utterance'].values.tolist(), df['next_utterance'].values.tolist())


    GRAPH.meta_data = df[meta_data_cols].to_dict(orient='records')
    GRAPH.checkpoint(output_name_)
    print()

print('=======][=======\n')
print(dt.now()-start)
