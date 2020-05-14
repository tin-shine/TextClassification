import os
import torch
import jieba
import torch.nn as nn
from config import Config
from torchtext import data
from torchtext.vocab import Vectors, GloVe
from sklearn.metrics import precision_recall_fscore_support
from torchtext.data import LabelField, Field, TabularDataset, BucketIterator

# 文本预处理
def pre_process_text():

    ID = Field(sequential=False, use_vocab=False)
    # 处理CATEGORY，标签选择非序列，use_vocab置true建立词典，is_target置true指明这是目标变量
    CATEGORY = LabelField(sequential=False, use_vocab=True, is_target=True)
    # 处理NEWS，文本选择序列，分词函数用jieba的lcut，返回句子原始长度方便RNN使用
    NEWS = Field(sequential=True, tokenize=jieba.lcut, include_lengths=True)

    fields = [
        ('id', ID),
        (None, None),
        ('category', CATEGORY),
        ('news', NEWS),
    ]

    # 加载数据
    train_data = TabularDataset(
        os.path.join('data', 'train.csv'),
        format = 'csv',
        fields = fields,
        csv_reader_params={'delimiter': '\t'}
    )    
    valid_data = TabularDataset(
        os.path.join('data', 'dev.csv'),
        format = 'csv',
        fields = fields,
        csv_reader_params={'delimiter': '\t'}
    )    
    test_data = TabularDataset(
        os.path.join('data', 'test.csv'),
        format = 'csv',
        fields = fields,
        csv_reader_params={'delimiter': '\t'}
    )
    
    # 创建字典
    NEWS.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    CATEGORY.build_vocab(train_data)

    return CATEGORY, NEWS, train_data, valid_data, test_data

# 建立模型
class Classifier(nn.Module):
    def __init__(self, vocab_size, output_dim, embedding_dim=300, hidden_dim=128,
                n_layers=2, bidirectional=False, dropout=float(0.1), pad_idx=None):
        super().__init__()

        # embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # lstm层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers = n_layers,
            bidirectional = bidirectional,
            dropout=dropout
        )

        # 全连接层
        num_direction = 2 if bidirectional else 1   # 双向
        self.fc = nn.Linear(hidden_dim * n_layers * num_direction, output_dim)
        # 丢弃概率
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, text, text_len):

        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_len)
        packed_output, (h_n, c_n) = self.lstm(packed_embedded)
        h_n = self.dropout(h_n)
        h_n = torch.transpose(h_n, 0, 1).contiguous()
        h_n = h_n.view(h_n.shape[0], -1)
        loggits = self.fc(h_n)

        return loggits

# 训练模型
def train(model, iterator, optimizer, scheduler, criterion, writer, step):
    # 将模型设为训练模式
    model.train()
    # 定义损失
    epoch_loss = 0
    epoch_acc = 0
    valN = 0

    for batch in iterator:
        # 为优化器设置0梯度
        optimizer.zero_grad()

        news, news_len = batch.news
        category = batch.category

        # 预测
        predictions = model(news, news_len)
        # 计算损失
        loss = criterion(predictions, category)
        # 反向传播损耗
        loss.backward()
        # 更新权重，更新学习率
        optimizer.step()
        scheduler.step()
        # 累计损失
        epoch_loss += loss.item()
        # 计算准确率
        preds = predictions.max(1)[1]
        epoch_acc += (preds==batch.category).sum().item()
        valN += batch.category.size(0)

        writer.add_scalar('Train/Loss', loss.item(), step)
        writer.add_scalar('Train/lr', scheduler.get_last_lr()[0], step)
        step += 1

    return epoch_loss / len(iterator), epoch_acc / valN, step
    
# 评估模型
def evaluate(model, iterator, criterion, writer, step):
    # 评估模式
    model.eval()
    # 初始化损失
    epoch_loss = 0
    epoch_acc = 0
    val_number = 0
    labels_list, preds_list = [], []
    with torch.no_grad():
        for batch in iterator:
            news, news_len = batch.news
            predictions = model(news, news_len)
            loss = criterion(predictions, batch.category)
            epoch_loss += loss
            preds = predictions.max(1)[1]
            epoch_acc += ((preds==batch.category).sum().item())
            val_number += batch.category.size(0)
            preds_list.append(preds)
            labels_list.append(batch.category)
            if (step > 0 and step % 50 == 0):
                y_true = torch.cat(labels_list).detach().cpu().numpy()
                y_pred = torch.cat(preds_list).detach().cpu().numpy()
                precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
                results = {
                    'loss': epoch_loss / step,
                    'f1': f1_score,
                    'precision': precision,
                    'recall': recall
                }
                for key, value in results.items():
                    writer.add_scalar("Eval/{}".format(key), value, step)
            step += 1
    return epoch_loss / len(iterator), epoch_acc / val_number, step

    
# 使用模型进行预测
def predict(model, iterator, criterion, writer):
    # 评估模式
    model.eval()
    # 初始化损失
    epoch_loss = 0
    epoch_acc = 0
    val_number = 0
    with torch.no_grad():
        for batch in iterator:
            news, news_len = batch.news
            predictions = model(news, news_len)
            loss = criterion(predictions, batch.category)
            epoch_loss += loss
            preds = predictions.max(1)[1]
            epoch_acc += ((preds==batch.category).sum().item())
            val_number += batch.category.size(0)

    return epoch_loss / len(iterator), epoch_acc / val_number