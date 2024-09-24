# Fine-Tuning BERT for Turkish Legal Text Processing

This repository demonstrates how to fine-tune a BERT model for masked language modeling (MLM) using Turkish legal documents in `.doc` and `.docx` formats. The code is designed to read legal text files, tokenize the sentences using a BERT tokenizer, and fine-tune the BERT model on the legal dataset.

### Key Features
- **Document Reading**:
  - The code reads `.doc` and `.docx` files from a directory and tokenizes the text into sentences using NLTK's Turkish sentence tokenizer.

- **BERT Model for Turkish**:
  - The pre-trained BERT model (`dbmdz/bert-base-turkish-cased`) is used for tokenization and masked language modeling.
  - The model is fine-tuned using legal texts, making it more suitable for legal language understanding tasks.

- **Custom Dataset and DataLoader**:
  - A custom PyTorch `Dataset` class is created to store tokenized sentences.
  - A custom `collate_fn` is implemented to handle token padding within batches for varying sequence lengths.
  - A `DataLoader` is used to load the dataset into batches for training.

- **Training and Optimization**:
  - The model is trained for 3 epochs using **AdamW** optimizer with a learning rate of `5e-5`.
  - Each batch of tokenized data is passed through the BERT model, and the masked language model (MLM) loss is computed.
  - The fine-tuned model is saved after training for further use.

### Requirements
- Python 3.x
- PyTorch
- Hugging Face Transformers
- NLTK
- `docx2txt` for reading `.docx` and `.doc` files

### Usage
1. Place your Turkish legal documents (in `.docx` or `.doc` format) in the `yasalar/` directory.
2. Run the provided script to fine-tune the BERT model on your legal text data.
3. After training, the fine-tuned model will be saved as `fine_tuned_bert_legal`.

### Output
- **Fine-tuned BERT Model**: Saved in the `fine_tuned_bert_legal/` directory, ready for downstream legal language processing tasks.

### License
This project is licensed under the MIT License.
