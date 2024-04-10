# custom_dataset.py
from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'article_input_ids': torch.tensor(self.data.iloc[idx]['article']['input_ids']),
            'article_attention_mask': torch.tensor(self.data.iloc[idx]['article']['attention_mask']),
            'summary_input_ids': torch.tensor(self.data.iloc[idx]['summary']['input_ids']),
            'summary_attention_mask': torch.tensor(self.data.iloc[idx]['summary']['attention_mask'])
        }
