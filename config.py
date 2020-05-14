import torch

class Config:

    data_dir = '.data'
    train_dir = 'train.csv'
    valid_dir = 'dev.csv'
    test_dir = 'test.csv'

    vocab_size = 400000 # 单词表容量
    num_labels = 5
    dropout = 0.1
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_EPOCHS = 10
    learning_rate = 1e-4
    