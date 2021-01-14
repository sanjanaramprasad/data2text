import sys
sys.path.append('..')
import logging
logging.getLogger().setLevel(100)
from fastprogress import progress_bar
from fastai.basics import Transform, Datasets, RandomSplitter, Module, Learner, ranger, params, load_learner
from fastai.text.all import TensorText
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer, BartTokenizer, BartForConditionalGeneration, BartConfig 
import torch
from torch.nn import functional as F
from torch import nn

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class DataTransform(Transform):
    def __init__(self, tokenizer:PreTrainedTokenizer, column:str):
        self.tokenizer = tokenizer
        self.column = column
        
    def encodes(self, inp):  
        #print(list(inp[self.column]))
        tokenized = self.tokenizer.batch_encode_plus(
            list(inp[self.column]),
            max_length=args.max_seq_len, 
            pad_to_max_length=True, 
            return_tensors='pt'
        )
        return TensorText(tokenized['input_ids']).squeeze()
        
    def decodes(self, encoded):
        decoded = [
            self.tokenizer.decode(
                o, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            ) for o in encoded
        ]
        return decoded

def load_hf_model(config, pretrained=False, path=None): 
    if pretrained:    
        if path:
            model = BartForConditionalGeneration.from_pretrained(
                "facebook/bart-large-cnn", 
                state_dict=torch.load(path, map_location=torch.device(args.device)), 
                config=config
            )
        else: 
            model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", config=config)
    else:
        model = BartForConditionalGeneration()

    return model.to(args.device)

class FastaiWrapper(Module):
    def __init__(self):
        self.config = BartConfig(vocab_size=50264, output_past=True)
        self.bart = load_hf_model(config=self.config, pretrained=True)
        
    def forward(self, x):
        output = self.bart(x)[0]
        return output

class SummarisationLoss(Module):
    def __init__(self):
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, output, target):
        x = F.log_softmax(output, dim=-1)
        norm = (target != 1).data.sum()
        return self.criterion(x.contiguous().view(-1, x.size(-1)), target.contiguous().view(-1)) / norm


def bart_splitter(model):
    return [
        params(model.bart.model.encoder), 
        params(model.bart.model.decoder.embed_tokens),
        params(model.bart.model.decoder.embed_positions),
        params(model.bart.model.decoder.layers),
        params(model.bart.model.decoder.layernorm_embedding),
    ]




args = Namespace(
    batch_size=4,
    max_seq_len=20000,
    data_path="/Users/sanjana/destruct/destruct/data/robo_train.csv",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), # ('cpu'),
    stories_folder='../data/my_own_stories',
    subset=None,
    test_pct=0.1
)
train_ds = pd.read_csv(args.data_path)
valid_ds = pd.read_csv('/Users/sanjana/destruct/destruct/data/robo_dev.csv')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn', add_prefix_space=True)

x_tfms = [DataTransform(tokenizer, column='source')]
y_tfms = [DataTransform(tokenizer, column='target')]
dss = Datasets(
    train_ds, 
    tfms=[x_tfms, y_tfms], 
    splits=RandomSplitter(valid_pct=0.1)(range(train_ds.shape[0]))
)
dls = dss.dataloaders(bs=args.batch_size, device=args.device.type)

learn = Learner(
    dls, 
    FastaiWrapper(), 
    loss_func=SummarisationLoss(), 
    opt_func=ranger,
    splitter=bart_splitter
)#.to_fp16()

learn.fit_flat_cos(
    1,
    lr=1e-4
)

learn.freeze_to(-2)
learn.dls.train.bs = args.batch_size//2
learn.dls.valid.bs = args.batch_size//2

learn.lr_find()
learn.fit_flat_cos(
    2,
    lr=1e-5
)
learn.export('../models/fintuned_bart.pkl')
