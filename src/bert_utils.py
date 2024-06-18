import string
from torch.utils.data import Dataset

VOCAB = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
         'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
         'X', 'B', 'U', 'Z', 'O']

# Simple character-level tokenizer
class CharTokenizer:
    def __init__(self):
        self.char2idx = {char: idx for idx, char in enumerate(VOCAB)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.pad_token_id = self._add_special_token('<PAD>')
        self.unk_token_id = self._add_special_token('<UNK>')
        self.cls_token_id = self._add_special_token('<CLS>')
        self.sep_token_id = self._add_special_token('<EOS>')
        self.mask_token_id = self._add_special_token('<MASK>')

    def _add_special_token(self, token):
        token_id = len(self.char2idx)
        self.char2idx[token] = token_id
        self.idx2char[token_id] = token
        return token_id

    def encode(self, protein_sequence):
        amino_acid_ids = [self.char2idx.get(char, self.unk_token_id) for char in protein_sequence.upper()]

    def decode(self, tokens):
        return ''.join([self.idx2char.get(token, '') for token in tokens])

    def pad(self, tokens, max_length):
        return tokens + [self.pad_token_id] * (max_length - len(tokens))


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {'input_ids': self.encodings[idx], 'labels': self.labels[idx]}

    def __len__(self):
        return len(self.labels)
