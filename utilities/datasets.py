from torch.utils.data import Dataset

class SID_dataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        input = record['body']
        label = record['topic']
        return input, label