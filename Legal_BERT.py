import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from torch.utils.data import Dataset, DataLoader
import os
from docx import Document
from nltk.tokenize import sent_tokenize
import docx2txt
import nltk
nltk.download('punkt')

def read_word_files(directory_path):
    sentences = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(directory_path, filename)
            text = docx2txt.process(file_path)
            para_sentences = sent_tokenize(text, language='turkish')
            sentences.extend(para_sentences)
        elif filename.endswith(".doc"):
            file_path = os.path.join(directory_path, filename)
            text = docx2txt.process(file_path)
            para_sentences = sent_tokenize(text, language='turkish')
            sentences.extend(para_sentences)

    return sentences

tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
model = BertForMaskedLM.from_pretrained("dbmdz/bert-base-turkish-cased")

directory_path = 'yasalar'

law_sentences = read_word_files(directory_path)

tokenized_text = [tokenizer.tokenize(sentence) for sentence in law_sentences]

from torch.nn.utils.rnn import pad_sequence


class LegalDataset(Dataset):
    def __init__(self, tokenized_text, tokenizer):
        self.tokenized_text = tokenized_text
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tokenized_text)

    def __getitem__(self, idx):
        return self.tokenized_text[idx]

    def collate_fn(self, batch):
        """
        Custom collate function to pad sequences within each batch.
        """
        tokens = [self.tokenizer.encode(sentence, add_special_tokens=True) for sentence in batch]

        padded_tokens = pad_sequence([torch.tensor(token) for token in tokens], batch_first=True)

        return padded_tokens


legal_dataset = LegalDataset(tokenized_text, tokenizer)
dataloader = DataLoader(legal_dataset, batch_size=4, shuffle=True, collate_fn=legal_dataset.collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(num_epochs):
    for batch in dataloader:
        batch = torch.tensor(batch).to(device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

model.save_pretrained("fine_tuned_bert_legal")
