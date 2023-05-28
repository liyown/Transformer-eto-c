import torch
from torchtext.data import get_tokenizer
from models.model import TranslationModel
from utils.utils import sentence_to_tensor, TranslationDataset, tensor_to_sentence

traindataset = TranslationDataset("./data/translation2019zh_valid.json")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = torch.load("model.pt")
model = model.eval()
tokenizer = get_tokenizer('basic_english')
print()
def translate(src: str):
    """
    :param src: 英文句子，例如 "I like machine learning."
    :return: 翻译后的句子，例如：”我喜欢机器学习“
    """

    # 将与原句子分词后，通过词典转为index，然后增加<bos>和<eos>
    src = torch.tensor([0] + sentence_to_tensor(tokenizer(src), traindataset.english_vocab) + [1]).unsqueeze(0).to(device)
    # 首次tgt为<bos>
    tgt = torch.tensor([[0]]).to(device)
    # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
    for i in range(50):
        # 进行transformer计算
        out = model(src, tgt)
        # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
        predict = model.predictor(out[:, -1])
        # 找出最大值的index
        y = torch.argmax(predict, dim=1)
        # 和之前的预测结果拼接到一起
        tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
        # 如果为<eos>，说明预测结束，跳出循环
        if y == 1:
            break
    # 将预测tokens拼起来
    tgt = tensor_to_sentence(tgt.squeeze(0), traindataset.chinese_vocab)
    return tgt


print(translate("The 12-second run was the fastest he has run since his injury four years ago."))