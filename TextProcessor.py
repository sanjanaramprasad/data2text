
import torch
import torchtext
import codecs
import io
from itertools import chain
from collections import Counter

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'

def extract_text_features(line):
    tokens = [token.split(u"￨") for token in line]
    tokens = [token for token in tokens if token[0]]
    row_size = len(tokens[0])

    words_features = list(zip(*tokens))
    words = words_features[0]
    features = words_features[1:]

    return words, features, row_size -1

class TextIterator(object):
    def __init__(self, corpus_path, line_truncate, side,
                 assoc_iter=None):
        try:
            # The codecs module seems to have bugs with seek()/tell(),
            # so we use io.open().
            self.corpus = io.open(corpus_path, "r", encoding="utf-8")
        except IOError:
            sys.stderr.write("Failed to open corpus file: %s" % corpus_path)
            sys.exit(1)
        #print(self.corpus)
        self.line_truncate = line_truncate
        self.side = side
        self.assoc_iter = assoc_iter
        self.last_pos = 0
        self.line_index = -1
        self.eof = False
    

    def __iter__(self):
        """
        Iterator of (example_dict, nfeats).
        On each call, it iterates over as many (example_dict, nfeats) tuples
        until this shard's size equals to or approximates `self.shard_size`.
        """
        print('in iteration')
        iteration_index = -1
        if self.assoc_iter is not None:
            # We have associate iterator, just yields tuples
            # util we run parallel with it.
            while self.line_index < self.assoc_iter.line_index:
                line = self.corpus.readline()
                if line == '':
                    raise AssertionError(
                        "Two corpuses must have same number of lines!")

                self.line_index += 1
                iteration_index += 1
                yield self._example_dict_iter(line, iteration_index)

            if self.assoc_iter.eof:
                self.eof = True
                self.corpus.close()
        else:
            # Yield tuples util this shard's size reaches the threshold.
            #print(self.corpus.seek(self.last_pos))
            print('in else')
            flag = True
            while flag:
                line = self.corpus.readline()
                if line == '':
                    self.eof = True
                    print('END line')
                    self.corpus.close()
                    break

                self.line_index += 1
                iteration_index += 1
                yield self._example_dict_iter(line, iteration_index)

    def hit_end(self):
        return self.eof

    @property
    def num_feats(self):
        # We peek the first line and seek back to
        # the beginning of the file.
        saved_pos = self.corpus.tell()

        line = self.corpus.readline().split()
        if self.line_truncate:
            line = line[:self.line_truncate]
        _, _, self.n_feats = extract_text_features(line)

        self.corpus.seek(saved_pos)

        return self.n_feats

    def _example_dict_iter(self, line, index):
        line = line.split()
        if self.line_truncate:
            line = line[:self.line_truncate]
        words, feats, n_feats = extract_text_features(line)
        if self.side == "src":
            entities_len = [22] * 26
            entities_len.extend([15, 15])
            entities_list = []
            for i in range(0, 28):
                if i < 26:
                    entities_list.extend([i] * 22)
                else:
                    entities_list.extend([i] * 15)
            count_entities = 28
            assert len(entities_len) == len(set(entities_list))
            example_dict = {self.side: words, "entities_list": entities_list, "entities_len": entities_len,
                            "count_entities": count_entities, "total_entities_list": entities_list,
                            "indices": index}
        else:
            example_dict = {self.side: words, "indices": index}
        if feats:
            # All examples must have same number of features.
            #aeq(self.n_feats, n_feats)

            prefix = self.side + "_feat_"
            example_dict.update((prefix + str(j), f)
                                for j, f in enumerate(feats))
        #print(example_dict)
        return example_dict

class DatsetProcessor(torchtext.data.Dataset):
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def load_fields(self, vocab_dict):
        from io_utils import load_fields_from_vocab

        fields = load_fields_from_vocab(vocab_dict.items(), self.data_type)
        self.fields = dict([(k, f) for (k, f) in fields.items()
                           if k in self.examples[0].__dict__])

    @staticmethod
    def extract_text_features(tokens):
        
        if not tokens:
            return [], [], -1

        split_tokens = [token.split(u"￨") for token in tokens]
        split_tokens = [token for token in split_tokens if token[0]]
        token_size = len(split_tokens[0])

        assert all(len(token) == token_size for token in split_tokens), \
            "all words must have the same number of features"
        words_and_features = list(zip(*split_tokens))
        words = words_and_features[0]
        features = words_and_features[1:]

        return words, features, token_size - 1
    
    # Below are helper functions for intra-class use only.

    def _join_dicts(self, *args):
        
        return dict(chain(*[d.items() for d in args]))

    def _peek(self, seq):
        
        first = next(seq)
        return first, chain([first], seq)

    def _construct_example_fromlist(self, data, fields):
       
        ex = torchtext.data.Example()
        for (name, field), val in zip(fields, data):
            if field is not None:
                setattr(ex, name, field.preprocess(val))
            else: 
                setattr(ex, name, val)
        return ex


class TextDataset(DatsetProcessor):
    
    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 num_src_feats=0, num_tgt_feats=0,
                 src_seq_length=0, tgt_seq_length=0,
                 dynamic_dict=True, use_filter_pred=True):
        self.data_type = 'text'

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.
        if tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))

        else:
            examples_iter = src_examples_iter

        if dynamic_dict:
            examples_iter = self._dynamic_dict(examples_iter)

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        src_size = 0

        out_examples = []
        for ex_values in example_values:
            example = self._construct_example_fromlist(
                ex_values, out_fields)
            src_size += len(example.src)
            out_examples.append(example)

        print("average src size", src_size / len(out_examples),
              len(out_examples))

        def filter_pred(example):
            return 0 < len(example.src) <= src_seq_length \
               and 0 < len(example.tgt) <= tgt_seq_length

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(TextDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    def _dynamic_dict(self, examples_iter):
        for example in examples_iter:
            src = example["src"]
            src_vocab = torchtext.vocab.Vocab(Counter(src),
                                              specials=[UNK_WORD, PAD_WORD])
            self.src_vocabs.append(src_vocab)
            # Mapping source tokens to indices in the dynamic dict.
            src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
            example["src_map"] = src_map

            example["entities_list"] = torch.LongTensor(example["entities_list"])
            example["entities_len"] = torch.LongTensor(example["entities_len"])
            example["total_entities_list"] = torch.LongTensor(example["total_entities_list"])
            if "tgt" in example:
                tgt = example["tgt"]
                mask = torch.LongTensor(
                    [0] + [src_vocab.stoi[w] for w in tgt] + [0])
                example["alignment"] = mask
            yield example

    def _peek(self, seq):
        
        first = next(seq)
        return first, chain([first], seq)

    def _join_dicts(self, *args):
        
        return dict(chain(*[d.items() for d in args]))

    def _construct_example_fromlist(self, data, fields):
        
        ex = torchtext.data.Example()
        for (name, field), val in zip(fields, data):
            if field is not None:
                setattr(ex, name, field.preprocess(val))
            else:
                setattr(ex, name, val)
        return ex


class TextProcessor:
    def __init__(self):
        return

    def get_num_features(self, data_file, input_type):
        with codecs.open(data_file, "r", "utf-8") as cf:
            f_line = cf.readline().strip().split()
            words, features, num_features = extract_text_features(f_line)

        return num_features


    def get_fields(self, num_source_features, num_target_features):

        ''' local functions needed '''

        def make_entities(data, vocab, is_train):
            source_size = max([r.size(0) for r in data])
            entities_size = max([r.max() for r in data]) + 1
            entities_mapping = torch.zeros(source_size, len(data), entities_size)
            for i, row in enumerate(data):
                for j, column in enumerate(row):
                    entities_mapping[j, i, column] = 1
            return entities_mapping

        
        def make_entities_len(data, vocab, is_trained):
            source_size = max([t.size(0) for t in data])
            entities_len_mapping = torch.ones(source_size, len(data))
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    entities_len_mapping[j][i] = t
            return entities_len_mapping

        def make_total_entities(data, vocab, is_train):
            source_size = max([t.size(0) for t in data])
            entities_size = max([t.max() for t in data]) + 1
            entities_mapping = torch.zeros(entities_size, len(data), source_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    entities_mapping[t, i, j] = 1
            return entities_mapping

        def make_src(data, vocab, is_train):
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(src_size, len(data), src_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        def make_tgt(data, vocab, is_train):
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[:sent.size(0), i] = sent
            return alignment


        ''' main function flows starts here '''

        fields = {}

        #TODO : implement diff torchtext class from Field class 
        fields['src'] = torchtext.data.Field(pad_token = PAD_WORD, include_lengths = True)

        for j in range(num_source_features):
            fields["src_feat_"+str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

        '''fields['entities_list'] = torchtext.data.Field(
            use_vocab=False, 
            postprocessing=make_entities, sequential=False)


        fields["entities_len"] = torchtext.data.Field(
            use_vocab=False,
            postprocessing=make_entities_len, sequential=False)

        fields["count_entities"] = torchtext.data.Field(
            use_vocab=False, 
            sequential=False)

        fields["total_entities_list"] = torchtext.data.Field(
            use_vocab=False, 
            postprocessing=make_total_entities, sequential=False)'''

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)
        
        for j in range(num_target_features):
            fields["tgt_feat_"+str(j)] = \
                torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD,
                                     pad_token=PAD_WORD)

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, 
            postprocessing=make_src, sequential=False)

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, 
            postprocessing=make_tgt, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False,
            sequential=False)
        
        return fields


def preprocess(text):
    text = re.sub(r'[^\w\s]',' ',text)
    text = text.lower()
    text = ' '.join([each.strip() for each in text.strip().split() if each.strip()])
    return text

def get_rr_data(review):
        review_summ = []
        review_summ.append(preprocess(review['sample_size']) )
        if review['interventions']:
            interventions = [preprocess(each) for each in review['interventions']]
            interventions_data_ind = [len(each) for each in interventions]
            max_len = max(interventions_data_ind)
            int_data = interventions[interventions_data_ind.index(max_len)]
        else:
            int_data = 'N/A'
        review_summ.append(int_data)

        outcomes = ' '.join([preprocess(each) for each in review['outcomes']])
        review_summ.append(outcomes)
        review_summ.append(preprocess(review['punchline_text']))
        review_summ.append(preprocess(review['random_sequence_generation']['judgement']))
        review_summ.append(preprocess(review['allocation_concealment']['judgement']))
        review_summ.append(preprocess(review['blinding_participants_personnel']['judgement']))
        #print(review_summ)
        #src_summaries.append('|'.join(review_summ))
        return '|'.join(review_summ)

def make_tabular_data(rr_data, abs_sum):
    src_data = []
    tgt_data = []
    for key, rid in abs_sum['ReviewID'].items():
        abs_summaries = abs_sum['Abstract'][key]
        rr_extractions = rr_data[rid]

        for abs_summary, rr_datapoint in list(zip(abs_summaries, rr_extractions)):
            if abs_summary:
                abs_summary = preprocess(abs_summary)
                src_data.append(get_rr_data(rr_datapoint))
                tgt_data.append(abs_summary)
                            
    return src_data, tgt_data


def write_data(src_data, tgt_data, src_file, tgt_file):
    
    
    with open(src_file, 'w') as fp1:
        fp1.write('\n'.join(src_data))
    
        
    with open(tgt_file, 'w') as fp2:
        fp2.write('\n'.join(tgt_data))

'''
data_path = '/Users/sanjana/destruct/destruct/data/roboreviewer'
with open(data_path + '/RR-dev.json', 'r') as fp:
    rr_data = json.load(fp)
    
with open(data_path + '/abstracts-summarization-dev.json', 'r') as fp:
    abs_sum = json.load(fp)



'''