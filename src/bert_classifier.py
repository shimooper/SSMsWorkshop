import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import nn
from transformers import BertConfig, BertModel
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb

from src.bert_utils import CharTokenizer, CustomDataset

WANDB_KEY = "64c3807b305e96e26550193f5860452b88d85999"
WANDB_PROJECT = "bert_classifier_signal_peptide"
RUN_NAME = "3-epochs"

REPO_BASE_PATH = Path(r'C:\repos\SSMsWorkshop')
SIGNAL_PEPTIDE_DIR_PATH = REPO_BASE_PATH / 'benchmarks' / 'signal_peptide'
SIGNAL_PEPTIDE_MODELS = REPO_BASE_PATH / 'models' / 'signal_peptide'


def tokenize_data(tokenizer, df, max_length):
    encodings = []
    attention_masks = []
    for sequence in df['seq']:
        tokens, mask = tokenizer.pad(tokenizer.encode(sequence), max_length)
        encodings.append(tokens)
        attention_masks.append(mask)

    labels = df['label'].tolist()
    return torch.tensor(encodings), torch.tensor(attention_masks), torch.tensor(labels)


# Define the PyTorch Lightning model
class BertForBinaryClassification(L.LightningModule):
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
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask=attention_mask)
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
    val_df = pd.read_csv(SIGNAL_PEPTIDE_DIR_PATH / 'valid_split.csv')
    test_df = pd.read_csv(SIGNAL_PEPTIDE_DIR_PATH / 'test.csv')

    # Tokenize and encode the dataset
    tokenizer = CharTokenizer()

    max_length = 72  # Maximum length of text sequences
    train_encodings, train_attention_masks, train_labels = tokenize_data(tokenizer, train_df, max_length)
    val_encodings, val_attention_masks, val_labels = tokenize_data(tokenizer, val_df, max_length)
    test_encodings, test_attention_masks, test_labels = tokenize_data(tokenizer, test_df, max_length)

    # Create Dataset objects
    train_dataset = CustomDataset(train_encodings, train_attention_masks, train_labels)
    val_dataset = CustomDataset(val_encodings, val_attention_masks, val_labels)
    test_dataset = CustomDataset(test_encodings, test_attention_masks, test_labels)

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

    # Logging with Weights & Biases
    wandb.login(key=WANDB_KEY)
    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=RUN_NAME)

    # Init ModelCheckpoint callback, monitoring "val_loss"
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Defines to metric to monitor and save checkpoints according to it
        save_top_k=1  # k - saves the k best model according to 'monitor'. None (default) - saves a checkpoint only for the last epoch, -1 = saves all checkpoints.
    )

    # Define the PyTorch Lightning Trainer
    trainer = L.Trainer(
        max_epochs=3,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Save the model
    torch.save(model.state_dict(), SIGNAL_PEPTIDE_MODELS / "bert_signal_peptide.pth")

    # Evaluate the model on the test dataset
    trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
