from entity import pre_process_text, Classifier, train, evaluate, predict
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import BucketIterator
from torch.optim import Adam, lr_scheduler
from config import Config
import torch.nn as nn
import torch

# 文本预处理
CATEGORY, NEWS, train_data, valid_data, test_data = pre_process_text('train')

# 建立模型
model = Classifier(
    vocab_size=len(NEWS.vocab),
    output_dim=Config.num_labels,
    pad_idx=NEWS.vocab.stoi[NEWS.pad_token],
    bidirectional=True,
    dropout=Config.dropout
)

# 将预训练词向量加载到embedding中
model.embedding.weight.data.copy_(NEWS.vocab.vectors)

# 以最小填充代价形成批，将长度相近的文本数据放在一批中 
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=32,
    sort_within_batch=True,
    sort_key = lambda x:len(x.news),
    device=torch.device('cpu')
)

# 优化器
optimizer = Adam(model.parameters(), Config.learning_rate)
# 调度器，用于调整优化器的学习率
scheduler = lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr = Config.learning_rate, 
    epochs = Config.N_EPOCHS,
    steps_per_epoch = len(train_iterator)
)
# 损失
criterion = nn.CrossEntropyLoss().to(Config.device)
# 将model所有参数梯度设置为0
model.zero_grad()

# 初始化最佳模型损失
best_loss = float('inf')

# 记录变量用于tensorboard可视化
writer = SummaryWriter()

# 训练模型
train_step = 0
eval_step = 0
for epoch in range(Config.N_EPOCHS):
    train_loss, train_acc, train_step = train(model, train_iterator, optimizer, scheduler, criterion, writer, train_step)
    valid_loss, valid_acc, eval_step = evaluate(model, valid_iterator, criterion, writer, eval_step)
    print("Epoch[", epoch, "]")
    print("train_loss=", train_loss, "train_acc=", train_acc)
    print("valid_loss=", valid_loss, "valid_acc=", valid_acc)
    
    if (best_loss > valid_loss):
        best_loss = valid_loss
        torch.save(model.state_dict(), 'save/optimal_model.pt')
        torch.save(optimizer.state_dict(), 'save/optimal_optimizer.pt')
        torch.save(scheduler.state_dict(), 'save/optimal_scheduler.pt')

writer.close()

# 测试模型
test_loss, test_acc = predict(model, test_iterator, criterion, writer)
test_acc *= 100
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')