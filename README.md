# NLP：文本多分类

## 目标

对给定文本进行多标签分类，使用给定数据，对文本分类模型进行训练、评估和预测，熟悉pytorch，torchtext，tensorflow等开发环境和工具，了解神经网络的基本工作原理。

## 环境

* Anaconda 2020.02
* python 3.7.6
* pytorch 1.5.0
* torchtext 0.6.0
* jieba 0.42.1
* sklearn 0.0
* tensorboard 2.2.1
* jupyter 1.0.0

## 内容

1. 给定数据集，已经划分成为三部分，分别为训练集``train.csv``，验证集``dev.csv``和测试集``test.csv``
2. 数据集中共有四列，第一列为数据``id``，第二列无意义，第三列为目标标签``caterogy``，共有五种标签，第四列为待分类文本``news``
3. 要求将文本分成五类

## 步骤详解

### 导入环境

```python
import os
import torch
import jieba
import torch.nn as nn
from torchtext import data
from torchtext.vocab import Vectors, GloVe
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support
from torchtext.data import LabelField, Field, TabularDataset, BucketIterator
```

## 预定义变量


```python
class Config:

    vocab_size = 400000 # 单词表容量
    num_labels = 5      # 目标标签数
    dropout = 0.1       # 默认丢弃率
    batch_size = 64     # 默认数据批大小
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设置CUDA
    N_EPOCHS = 10       # 设置模型训练次数
    learning_rate = 1e-4  # 初始学习率
    
```

### 文本预处理


```python
def pre_process_text():
    
    # ID无用
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
    # 会自动下载此词向量，大小约3.35GB
    NEWS.build_vocab(train_data, vectors=GloVe(name='6B', dim=300)) 
    CATEGORY.build_vocab(train_data)

    return CATEGORY, NEWS, train_data, valid_data, test_data

CATEGORY, NEWS, train_data, valid_data, test_data = pre_process_text()
```

## 构建模型

```python
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
        num_direction = 2 if bidirectional else 1   # 双向&单向可选
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
    
model = Classifier(
    vocab_size=len(NEWS.vocab),
    output_dim=Config.num_labels,
    pad_idx=NEWS.vocab.stoi[NEWS.pad_token],
    bidirectional=True,
    dropout=Config.dropout
)

```

## 加载预训练词向量

```python
model.embedding.weight.data.copy_(NEWS.vocab.vectors)
```

## 数据分批

```python
# 以最小填充代价形成批，将长度相近的文本数据放在一批中，一次性处理三个数据集
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=32,
    sort_within_batch=True,
    sort_key = lambda x:len(x.news),
    device=torch.device('cpu')
)
```

## 定义优化器


```python
# 使用Adam优化器
optimizer = Adam(model.parameters(), Config.learning_rate)
```

## 定义调度器


```python
# 用于调整优化器的学习率
scheduler = lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr = Config.learning_rate, 
    epochs = Config.N_EPOCHS,
    steps_per_epoch = len(train_iterator)
)
```

## 定义损失函数


```python
criterion = nn.CrossEntropyLoss().to(Config.device)
```

## 初始化训练参数

```python
# 将model所有参数梯度设置为0
model.zero_grad()

# 初始化最佳模型损失，用于比较保存最佳模型
best_loss = float('inf')

# 记录变量用于tensorboard可视化
writer = SummaryWriter()

# 分别用于训练和评估函数中，跨越两个循环，统计总运行批数据批次，用于组成可视化图表的横轴
train_step = 0
eval_step = 0
```

## 训练模型

### 训练方法


```python
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
        # 记录损失和学习率
        writer.add_scalar('Train/Loss', loss.item(), step)
        writer.add_scalar('Train/lr', scheduler.get_last_lr()[0], step)
        # 全局计数器自增
        step += 1
    # 返回损失率，正确率和全局计数器
    return epoch_loss / len(iterator), epoch_acc / valN, step
```

### 评估方法


```python
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
             # 每50步记录一次评估指标，使用precision，recall和f1 score评估当前模型性能
            if (step > 0 and step % 50 == 0): 
                y_true = torch.cat(labels_list).detach().cpu().numpy()
                y_pred = torch.cat(preds_list).detach().cpu().numpy()
                precision, recall, f1_score, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='micro'
                )
                results = { 'loss': epoch_loss / step, 'f1': f1_score,
                    'precision': precision, 'recall': recall
                }            
                for key, value in results.items():
                    writer.add_scalar("Eval/{}".format(key), value, step)
            step += 1
return epoch_loss / len(iterator), epoch_acc / val_number, step
```

### 开始训练，默认训练次数是10次


```python
for epoch in range(Config.N_EPOCHS):
    train_loss, train_acc, train_step = train(model, train_iterator, optimizer, 
                                              scheduler, criterion, writer, train_step)
    valid_loss, valid_acc, eval_step = evaluate(model, valid_iterator, criterion, 
                                                writer, eval_step)
    print("Epoch[", epoch, "]")
    print("train_loss=", train_loss, "train_acc=", train_acc)
    print("valid_loss=", valid_loss, "valid_acc=", valid_acc)
    # 每次比较模型，记录下最好的那个模型，保存状态参数
    if (best_loss > valid_loss):
        best_loss = valid_loss
        torch.save(model.state_dict(), 'save/optimal_model.pt')
        torch.save(optimizer.state_dict(), 'save/optimal_optimizer.pt')
        torch.save(scheduler.state_dict(), 'save/optimal_scheduler.pt')

# 训练结束后，关闭I/O
writer.close()
```

## 测试模型

### 定义预测方法


```python
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
```

### 预测


```python
# 使用最优模型对测试数据集进行预测，用准确率展示预测结果
test_loss, test_acc = predict(model, test_iterator, criterion, writer)
test_acc *= 100
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
```

## Tensorboard可视化

![pic1](https://i.loli.net/2020/05/14/AiwVUma7cFlbgeI.png)


## 参考文献及网站

【1】https://monkeylearn.com/text-classification/

【2】https://github.com/Yan2013/NLP2020/tree/classification

【3】https://zhuanlan.zhihu.com/p/94941514

【4】https://pytorch.org/docs/stable/index.html

【5】http://pytorchchina.com/

【6】https://pytorch.org/text/

【7】https://tuna.moe/
