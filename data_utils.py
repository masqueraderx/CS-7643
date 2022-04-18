import os
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from pytorch_transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences

def token2ids(text, MAX_LEN):
    sentences = []
    sentences.append("[CLS]" + text + "[SEP]")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    print(tokenized_texts)
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    return input_ids

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root # this includes the jsonl file, e.g.: hateful_memes/train.jsonl
        self.data = []
        self.transforms = transforms.Compose([transforms.ToTensor()])
        len_text = []
        f = open(root, 'r', encoding='utf-8')
        for line in f.readlines():
            self.data.append(json.loads(line))
            len_text.append(len(json.loads(line)['text']))
        self.max_len = max(len_text)

    def __getitem__(self, idx):
        img_path = os.path.join(os.path.dirname(self.root), self.data[idx]['img'])
        img = Image.open(img_path)
        img = self.transforms(img)

        input = {}
        text = self.data[idx]['text']
        input['text'] = self.transforms(token2ids(text, self.max_len))
        input['img'] = img

        target = {}
        target['label'] = torch.tensor([self.data[idx]['label']])
        target['image_id'] = torch.tensor([idx])
        return input, target

    def __len__(self):
        return len(self.data)



if __name__ == "__main__":
    root = 'hateful_memes/train.jsonl'

    # test MyDataset
    dataset = MyDataset(root)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    img_text, targets = next(iter(data_loader))
    print(img_text)
    print("\n")
    print(targets)

    # # test tokenize a sentence
    # f = open(root, 'r')
    # for line in f.readlines():
    #     data = json.loads(line)
    #     print(data['text'])
    #
    #     ids = token2ids(data['text'], 50)
    #     print(ids)
    #     break