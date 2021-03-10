import datasets
import pandas as pd
from fastai.text.all import *
from transformers import *

from blurr.data.all import *
from blurr.modeling.all import *

def read_csv(data_path):
    df = pd.read_csv(data_path)
    return df

pretrained_model_name = "facebook/bart-large"
hf_arch, hf_config, hf_tokenizer, hf_model = BLURR_MODEL_HELPER.get_hf_objects(pretrained_model_name, 
                                            model_cls=BartForConditionalGeneration)
hf_arch, type(hf_config), type(hf_tokenizer), type(hf_model)
text_gen_kwargs = default_text_gen_kwargs(hf_config, hf_model, task='summarization'); text_gen_kwargs

hf_batch_tfm = HF_Seq2SeqBeforeBatchTransform(hf_arch, hf_config, hf_tokenizer, hf_model, 
                                              max_length=256, max_tgt_length=130, text_gen_kwargs=text_gen_kwargs)

blocks = (HF_Seq2SeqBlock(before_batch_tfm=hf_batch_tfm), noop)
dblock = DataBlock((HF_Seq2SeqBlock(before_batch_tfm=hf_batch_tfm), noop), get_x=ColReader('source'), get_y=ColReader('target'), )

robo_df = read_csv('/home/sanjana/destruct/data/robo_train.csv')

dls = dblock.dataloaders(robo_df, bs = 2)

print(len(dls.train.items), len(dls.valid.items))

b = dls.one_batch()
print(len(b), b[0]['input_ids'].shape, b[1].shape)

seq2seq_metrics = {
        'rouge': {
            'compute_kwargs': { 'rouge_types': ["rouge1", "rouge2", "rougeL"], 'use_stemmer': True },
            'returns': ["rouge1", "rouge2", "rougeL"]
        },
        'bertscore': {
            'compute_kwargs': { 'lang': 'en' },
            'returns': ["precision", "recall", "f1"]
        }
    }

model = HF_BaseModelWrapper(hf_model)
learn_cbs = [HF_BaseModelCallback]
fit_cbs = [HF_Seq2SeqMetricsCallback(custom_metrics=seq2seq_metrics)]

learn = Learner(dls, 
                model,
                opt_func=ranger,
                loss_func=CrossEntropyLossFlat(),
                cbs=learn_cbs,
                splitter=partial(seq2seq_splitter, arch=hf_arch)).to_fp16()

learn.create_opt() 
learn.freeze()
learn.lr_find(suggestions=True)

b = dls.one_batch()
preds = learn.model(b[0])
print(len(preds),preds[0], preds[1].shape)
learn.fit_one_cycle(1, lr_max=3e-5, cbs=fit_cbs)

learn.show_results(learner=learn, max_n=2)

learn.metrics = None
learn.export(fname='rr_bart_export.pkl')
