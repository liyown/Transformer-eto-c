import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import TranslationModel
from utils.utils import collate_fn, TranslationLoss, TranslationDataset

traindataset = TranslationDataset("./data/translation2019zh_valid.json")

train_loader = DataLoader(traindataset, batch_size=50, shuffle=True, collate_fn=collate_fn)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = TranslationModel(512, traindataset.english_vocab, traindataset.chinese_vocab).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criteria = TranslationLoss()

model.train()
for epoch in range(1000):
    for index, data in enumerate(train_loader):
        # 生成数据
        src, tgt, tgt_y, n_tokens = data
        src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

        # 清空梯度
        optimizer.zero_grad()
        # 进行transformer的计算
        out = model(src, tgt)
        # 将结果送给最后的线性层进行预测
        out = model.predictor(out)

        """
        计算损失。由于训练时我们的是对所有的输出都进行预测，所以需要对out进行reshape一下。
                我们的out的Shape为(batch_size, 词数, 词典大小)，view之后变为：
                (batch_size*词数, 词典大小)。
                而在这些预测结果中，我们只需要对非<pad>部分进行，所以需要进行正则化。也就是
                除以n_tokens。
        """
        loss = criteria(out, tgt_y) / n_tokens
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()

        del src
        del tgt
        del tgt_y

    print(loss.item())

    torch.save(model, "model.pt")



