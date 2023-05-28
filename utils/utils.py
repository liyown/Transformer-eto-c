import json
from collections import Counter

import jieba
import h5py
import numpy as np
import torch
from torch import nn, tensor
from torch.nn.functional import pad, log_softmax
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from tqdm import tqdm


def read_file(json_path):
    english_sentences = []
    chinese_sentences = []
    tokenizer = get_tokenizer('basic_english')
    with open(json_path, 'r', encoding="utf-8") as fp:
        lines = fp.readlines()
        for line in lines:
            line = json.loads(line)
            english, chinese = line['english'], line['chinese']
            english = tokenizer(english)
            chinese = list(jieba.cut(chinese))
            english_sentences.append(english)
            chinese_sentences.append(chinese)
    return english_sentences, chinese_sentences


def create_vocab(sentences, max_element=None):
    """Note that max_element includes special characters"""

    default_list = ['<sos>', '<eos>', '<unk>', '<pad>']

    char_set = Counter()
    for sentence in tqdm(sentences):
        c_set = Counter(sentence)
        char_set.update(c_set)

    if max_element is None:
        return default_list + list(char_set.keys())
    else:
        max_element -= 4
        words_freq = char_set.most_common(max_element)
        # pair array to double array
        words, freq = zip(*words_freq)
        return default_list + list(words)


def sentence_to_tensor(sentences, vocab, UNK_ID=2):
    vocab_map = {k: i for i, k in enumerate(vocab)}
    res = list(map(vocab_map.get, sentences, [UNK_ID] * len(sentences)))
    return res


def tensor_to_sentence(sentences, vocab):
    vocab_map = {i: k for i, k in enumerate(vocab)}
    res = []
    for idx in sentences.tolist():
        res.append(vocab_map[idx])
    return "".join(res).replace("<sos>", "").replace("<eos>", "")


def collate_fn(batch, SOS_ID=0, EOS_ID=1, PAD_ID=3, max_len=50):
    src_list, tgt_list = [], []
    # 循环遍历句子对儿
    for (_src, _tgt) in batch:
        """
        _src: 英语句子，例如：`I love you`对应的index
        _tgt: 中文句子，例如：`我 爱 你`对应的index
        """
        processed_src = torch.cat(
            [torch.tensor([SOS_ID]), torch.tensor(_src, dtype=torch.int64), torch.tensor([EOS_ID])], 0)
        processed_tgt = torch.cat(
            [torch.tensor([SOS_ID]), torch.tensor(_tgt, dtype=torch.int64), torch.tensor([EOS_ID])], 0)

        src_list.append(pad(processed_src, (0, max_len - len(processed_src)), value=PAD_ID))
        tgt_list.append(pad(processed_tgt, (0, max_len - len(processed_tgt)), value=PAD_ID))

    # 将多个src句子堆叠到一起
    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)

    # tgt_y是目标句子去掉第一个token，即去掉<bos>
    tgt_y = tgt[:, 1:]
    # tgt是目标句子去掉最后一个token
    tgt = tgt[:, :-1]

    # 计算本次batch要预测的token数
    n_tokens = (tgt_y != PAD_ID).sum()

    # 返回batch后的结果
    return src, tgt, tgt_y, n_tokens


class TranslationLoss(nn.Module):

    def __init__(self, PAD_ID=3):
        super(TranslationLoss, self).__init__()
        # 使用KLDivLoss，不需要知道里面的具体细节。
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = PAD_ID

    def forward(self, x: tensor, target: tensor) -> tensor:
        dim = x.size()
        x = x.view(dim[0] * dim[1], -1)
        x = torch.nn.functional.log_softmax(x, 1)
        y = torch.zeros(dim[0] * dim[1], dim[2]).to("cuda:0")
        target = target.view(-1, 1)
        y.scatter_(1, target, 1)
        mask = torch.ones(dim[0] * dim[1], 1).to("cuda:0")
        mask[target == self.padding_idx] = 0
        y = y * mask

        return self.criterion(x, y.clone().detach())


class TranslationDataset(Dataset):
    def __init__(self, file_dir, evn=None, cvn=None):
        super(TranslationDataset, self).__init__()
        self.english_sentences, self.chinese_sentences = read_file(file_dir)
        self.english_vocab = create_vocab(self.english_sentences, max_element=evn)
        self.chinese_vocab = create_vocab(self.chinese_sentences, max_element=cvn)

    def __getitem__(self, index):
        english_tensor = sentence_to_tensor(self.english_sentences[index], self.english_vocab)
        chinese_tensor = sentence_to_tensor(self.chinese_sentences[index], self.chinese_vocab)
        return english_tensor, chinese_tensor

    def __len__(self):
        return len(self.english_sentences)


if __name__ == "__main__":
    data = TranslationDataset("../data/translation2019zh_valid.json")
    print(data[0])
    pass
