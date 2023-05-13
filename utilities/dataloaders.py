from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torch import tensor
import torch
from ast import literal_eval
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from utilities.utils import dict2w

# spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_thresh):
        self.freq_thresh = freq_thresh
        self.InS = dict2w()
        self.InS[0] = '<PAD>'
        self.InS[1] = '<SOS>'
        self.InS[2] = '<EOS>'
        self.InS[3] = '<UNK>'
        
        self.tokenizer = get_tokenizer('basic_english', language='en')

    def __len__(self):
        return len(self.InS)
    
    def tokenizer_eng(self, text):
        return self.tokenizer(text)
    
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = len(self.InS)
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else: frequencies[word] += 1
                if frequencies[word] == self.freq_thresh:
                    self.InS[word] = idx
                    idx += 1
    
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.InS[token] if token in self.InS else self.InS['<UNK>']
            for token in tokenized_text
        ]

        
class news_dataset(Dataset):
    def __init__(self, data, freq_thresh=5):
        self.body = data['body']
        self.topic = data['topic']

        self.vocab = Vocabulary(freq_thresh)
        self.vocab.build_vocabulary(self.body.tolist())
    
    def __len__(self):
        return len(self.body)
    
    def __getitem__(self, idx):
        label = literal_eval(self.topic.iloc[idx])
        label = tensor(label, dtype=torch.int64)

        text = self.body.iloc[idx]
        numerical_text = [self.vocab.InS['<SOS>']]
        numerical_text += self.vocab.numericalize(text)
        numerical_text.append(self.vocab.InS['<EOS>'])

        return tensor(numerical_text, dtype=torch.int64), label

class news_collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        labels = torch.cat([item[1].unsqueeze(0) for item in batch], dim=0).type(torch.float)
        texts = [item[0] for item in batch]
        texts = pad_sequence(texts, batch_first=True, padding_value=self.pad_idx)

        return texts, labels

def news_loader(data, batch_size=32, shuffle=True):
    dataset = news_dataset(data)
    vocab = dataset.vocab
    pad_idx = vocab.InS['<PAD>']

    mask = np.random.rand(len(data)) < .9
    train_data = data[mask]
    validation_data = data[~mask]

    train_dataset = news_dataset(train_data)

    val_dataset = news_dataset(validation_data)

    train_loader = DataLoader(dataset=train_dataset, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        collate_fn=news_collate(pad_idx=pad_idx))
    
    val_loader = DataLoader(dataset=val_dataset, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        collate_fn=news_collate(pad_idx=pad_idx))
    
    return vocab, train_loader, val_loader