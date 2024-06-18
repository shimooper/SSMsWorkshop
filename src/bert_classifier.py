import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import pytorch_lightning as pl
from torch import nn
from transformers import BertConfig, BertModel
from src.bert_utils import CharTokenizer, CustomDataset


SIGNAL_PEPTIDE_DIR_PATH = Path(r'C:\repos\SSMsWorkshop\benchmarks\signal_peptide')
SIGNAL_PEPTIDE_MODELS = Path(r'C:\repos\SSMsWorkshop\models\signal_peptide')


def tokenize_data(tokenizer, df, max_length):
    encodings = [tokenizer.encode(text) for text in df['seq']]
    encodings = [tokenizer.pad(enc, max_length) for enc in encodings]
    labels = df['label'].tolist()
    return torch.tensor(encodings), torch.tensor(labels)


# Define the PyTorch Lightning model
class BertForBinaryClassification(pl.LightningModule):
    def __init__(self, config):
        super(BertForBinaryClassification, self).__init__()
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 2)  # Binary classification
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        outputs = self(input_ids)
        loss = self.loss_fn(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        outputs = self(input_ids)
        loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        outputs = self(input_ids)
        loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer


def main():
    # Load your dataset
    train_df = pd.read_csv(SIGNAL_PEPTIDE_DIR_PATH / 'train_split.csv')
    val_df = pd.read_csv(SIGNAL_PEPTIDE_DIR_PATH / 'val_split.csv')
    test_df = pd.read_csv(SIGNAL_PEPTIDE_DIR_PATH / 'test.csv')

    # Tokenize and encode the dataset
    tokenizer = CharTokenizer()

    max_length = 70  # Maximum length of text sequences
    train_encodings, train_labels = tokenize_data(tokenizer, train_df, max_length)
    val_encodings, val_labels = tokenize_data(tokenizer, val_df, max_length)
    test_encodings, test_labels = tokenize_data(tokenizer, test_df, max_length)

    # Create Dataset objects
    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    # Define BERT configuration
    config = BertConfig(
        vocab_size=len(tokenizer.char2idx),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=max_length,
        pad_token_id=tokenizer.pad_token_id
    )

    # Instantiate the model
    model = BertForBinaryClassification(config)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Define the PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=3, gpus=1 if torch.cuda.is_available() else 0)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Save the model
    torch.save(model.state_dict(), SIGNAL_PEPTIDE_MODELS / "bert_signal_peptide.pth")

    # Evaluate the model on the test dataset
    trainer.test(model, test_loader)
