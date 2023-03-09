import pickle
import re

import torch
from transformers import AutoTokenizer

import functools
from collections import defaultdict


class FlotaTokenizer:
    def __init__(self, model, k, strict, mode):
        self.model = model
        self.k = k
        self.strict = strict
        self.mode = mode
        self.tok = AutoTokenizer.from_pretrained(model, model_max_length=512)
        self.test_data = {'input': defaultdict(list), 'output': defaultdict(list)}
        try:
            with open('vocabs/{}.p'.format(model), 'rb') as f:
                self.vocab = pickle.load(f)
        except FileNotFoundError:
            self.vocab = self.tok.get_vocab()
        assert len(self.vocab) == self.tok.vocab_size
        if self.model == 'bert-base-cased' or self.model == 'bert-base-uncased':
            self.special = '##'
            self.max_len = 18
        elif self.model == 'gpt2':
            self.special = '\u0120'
            self.max_len = 19
        elif self.model == 'xlnet-base-cased':
            self.special = '▁'
            self.max_len = 16

    def __call__(self, texts):
        self.test_data['input']['call'].append(texts)
        tensors = []

        texts = [self.encode(text) for text in texts]
        batch_size = len(texts)
        max_len = max(len(text) for text in texts)
        if self.model == 'bert-base-cased' or self.model == 'bert-base-uncased':
            input_ids = torch.zeros((batch_size, max_len)).long()
            attention_mask = torch.zeros((batch_size, max_len)).long()
            for i, text in enumerate(texts):
                input_ids[i, :len(text)] = torch.tensor(text)
                attention_mask[i, :len(text)] = 1
            tensors = {'input_ids': input_ids, 'attention_mask': attention_mask}
        elif self.model == 'gpt2':
            input_ids = self.tok.eos_token_id * torch.ones((batch_size, max_len)).long()
            attention_mask = torch.zeros((batch_size, max_len)).long()
            for i, text in enumerate(texts):
                input_ids[i, -len(text):] = torch.tensor(text)
                attention_mask[i, -len(text):] = 1
            tensors = {'input_ids': input_ids, 'attention_mask': attention_mask}
        elif self.model == 'xlnet-base-cased':
            input_ids = self.tok.pad_token_id * torch.ones((batch_size, max_len)).long()
            attention_mask = torch.zeros((batch_size, max_len)).long()
            for i, text in enumerate(texts):
                input_ids[i, -len(text):] = torch.tensor(text)
                attention_mask[i, -len(text):] = 1
            tensors = {'input_ids': input_ids, 'attention_mask': attention_mask}

        self.test_data['output']['call'].append(tensors)
        return tensors

    def max_subword_split(self, w):
        for l in range(min(len(w), self.max_len), 0, -1):
            for i in range(0, len(w) - l + 1):
                if w[i] == '-':
                    continue
                subword = w[i:i + l]
                if self.model == 'bert-base-cased' or self.model == 'bert-base-uncased':
                    if i == 0:
                        if subword in self.vocab:
                            return subword, w[:i] + l * '-' + w[i + l:], i
                        elif not self.strict and self.special + subword in self.vocab:
                            return self.special + subword, w[:i] + l * '-' + w[i + l:], i
                    else:
                        if self.special + subword in self. vocab:
                            return self.special + subword, w[:i] + l * '-' + w[i + l:], i
                        elif subword in self.vocab:
                            return subword, w[:i] + l * '-' + w[i + l:], i
                elif self.model == 'gpt2' or self.model == 'xlnet-base-cased':
                    if i == 0:
                        if self.special + subword in self.vocab:
                            return self.special + subword, w[:i] + l * '-' + w[i + l:], i
                        elif subword in self.vocab:
                            return subword, w[:i] + l * '-' + w[i + l:], i
                    else:
                        if subword in self.vocab:
                            return subword, w[:i] + l * '-' + w[i + l:], i
                        elif self.special + subword in self.vocab:
                            return self.special + subword, w[:i] + l * '-' + w[i + l:], i
        return None, None, None

    def get_flota_dict(self, w, k):
        max_subword, rest, i = self.max_subword_split(w)
        if max_subword is None:
            return dict()
        if k == 1 or rest == len(rest) * '-':
            flota_dict = {i: max_subword}
            return flota_dict
        flota_dict = self.get_flota_dict(rest, k - 1)
        flota_dict[i] = max_subword
        return flota_dict

    @functools.cache
    def tokenize(self, w):
        self.test_data['input']['tokenize'].append(w)
        return_value: list[str] | None = None

        if self.model == 'bert-base-cased' or self.model == 'bert-base-uncased':
            if w in self.vocab:
                return_value = [w]
            elif self.special + w in self.vocab:
                return_value = [self.special + w]
        elif self.model == 'gpt2' or self.model == 'xlnet-base-cased':
            if self.special + w in self.vocab:
                return_value = [self.special + w]
            elif w in self.vocab:
                return_value = [w]
        if return_value is not None:
            self.test_data['output']['tokenize'].append(return_value)
            return return_value

        if self.mode == 'flota':
            flota_dict = self.get_flota_dict(w, self.k)
            return_value = [subword for i, subword in sorted(flota_dict.items())]
        elif self.mode == 'first':
            if self.model == 'gpt2':
                return_value = self.tok.tokenize(' ' + w)[: self.k]
            else:
                return_value = self.tok.tokenize(w)[: self.k]
        elif self.mode == 'longest':
            if self.model == 'gpt2':
                subwords = enumerate(self.tok.tokenize(' ' + w))
            else:
                subwords = enumerate(self.tok.tokenize(w))
            topk_subwords = sorted(subwords, key=lambda x: len(x[1].lstrip(self.special)), reverse=True)[:self.k]
            return_value = [subword for i, subword in sorted(topk_subwords, key=lambda x: x[0])]

        self.test_data['output']['tokenize'].append(return_value)
        return return_value

    @functools.cache
    def encode(self, text):
        self.test_data['input']['encode'].append(text)
        encoded_text = []

        text_split = re.findall(r'[\w]+|[^\s\w]', text)
        tokens = list()
        for w in text_split:
            tokens.extend(self.tokenize(w))
        if self.model == 'bert-base-cased' or self.model == 'bert-base-uncased':
            ids_flota = self.tok.convert_tokens_to_ids(tokens)[: self.tok.model_max_length - 2]
            encoded_text = [self.tok.cls_token_id, *ids_flota] + [self.tok.sep_token_id]
        elif self.model == 'gpt2':
            encoded_text = self.tok.convert_tokens_to_ids(tokens)[: self.tok.model_max_length]
        elif self.model == 'xlnet-base-cased':
            ids_flota = self.tok.convert_tokens_to_ids(tokens)[: self.tok.model_max_length - 2 ]
            encoded_text = [*ids_flota, self.tok.sep_token_id, self.tok.cls_token_id]

        self.test_data['output']['encode'].append(encoded_text)
        return encoded_text
