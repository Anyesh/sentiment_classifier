from torchtext import data
from model import SentimentAnalysis
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import BertTokenizer
import random
import torch
import pandas as pd
from params import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    return tokens


TEXT = data.Field(batch_first=True,
                  use_vocab=False,
                  tokenize=tokenize_and_cut,
                  preprocessing=tokenizer.convert_tokens_to_ids,
                  init_token=init_token_idx,
                  eos_token=eos_token_idx,
                  pad_token=pad_token_idx,
                  unk_token=unk_token_idx)

LABEL = data.LabelField(dtype=torch.float)
df = pd.read_csv(
    f'{DATA_PATH}/tweets.csv')
temp_train_data = df[:int(0.8 * len(df))]  # 80% train
temp_train_data.to_csv(f'{DATA_PATH}/train.csv')
temp_test_data = df[int(0.8 * len(df)):]  # 20 % test
temp_test_data.to_csv(f'{DATA_PATH}/test.csv')


# init dataset and dataloader
print('initializing dataset...')
train_data = data.TabularDataset(path=f'{DATA_PATH}/train.csv', format='csv',  fields={
    'SentimentText': ('text', TEXT), 'Sentiment': ('labels', LABEL)})

test_data = data.TabularDataset(path=f'{DATA_PATH}/test.csv', format='csv',  fields={
    'SentimentText': ('text', TEXT), 'Sentiment': ('labels', LABEL)})

train_data, valid_data = train_data.split(random_state=random.seed(SEED))
print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")

# Building vocab
LABEL.build_vocab(train_data)

# Building iterator
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    sort=False,
    sort_within_batch=False,
    repeat=False)

print('dataloader initialized...')
