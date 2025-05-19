High-level steps:
1. Load data from HF datasets library
2. Tokenize the text columns
3. Format the tokenized data into tensors and apply attention masking.
4. Instantiate the appropriate dataset class, e.g. SFTDataset, DPODataset, RLOODataset
5. Instantiate a DataLoader with the appropriate dataset class.