# This script is used to create a dataset from a jsonl file for fine-tuning the OpenBuddy models.
# This script supports FastChat and OpenAI sample formats.

import torch
import os
import json
import random
import torch.multiprocessing as mp
from functools import partial
from tqdm import tqdm
from torch.utils.data import Dataset

# Must be the same value as the one defined in transformers.
IGNORE_INDEX = -100

NO_MASK = False

SEPS = ['\n', '\n\n', '、', '，' , '。', '；']


import hashlib
import os

PASS = ''
if 'PASS' in os.environ:
    PASS = os.environ['PASS']


def derive_key(password, salt):
    return hashlib.sha256(password.encode() + salt).digest()

def decrypt_file(filename, password):
    with open(filename, 'rb') as file:
        nonce = file.read(12)  
        ciphertext = file.read()

    key = derive_key(password, b"mysalt")

    cipher = ChaCha20.new(key=key, nonce=nonce)

    plaintext = cipher.decrypt(ciphertext)

    lines = plaintext.decode('utf-8').splitlines()

    return lines

class SupervisedDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=100, sample_format='fourfourml'):
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad token.")

        self.cached_items = {}
        self.data = []
        if file_path.lower().endswith('.enc'):
            global ChaCha20
            from Crypto.Cipher import ChaCha20
            lines = decrypt_file(file_path, PASS)
            for line in lines:
                item = json.loads(line)
                self.data.append(item)
        else:
            with open(file_path, 'r', encoding='utf8') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    self.data.append(item)
        self.bos_ids = tokenizer.encode("", add_special_tokens=True)
        self.role_assistant_says_ids = tokenizer.encode("<|role|>assistant<|says|>", add_special_tokens=False)
        self.nl_ids = tokenizer.encode('\n', add_special_tokens=False)
        assert len(self.nl_ids) == 1
        self.end_ids = tokenizer.encode('<|end|>', add_special_tokens=False)
        assert len(self.end_ids) == 1
        self.end_nl_ids = tokenizer.encode('<|end|>\n', add_special_tokens=False)
        assert self.end_nl_ids == self.end_ids + self.nl_ids
        self.sep_ids_set = set()
        for sep in SEPS:
            enc = tokenizer.encode(sep, add_special_tokens=False)
            self.sep_ids_set.add(enc[-1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print('__getitem__', index)
        if index in self.cached_items:
            return self.cached_items[index]
        
        dobj = self.data[index]

        input_ids = []
        labels = []
        input_ids += self.bos_ids
        labels += [IGNORE_INDEX] * len(self.bos_ids)
        if 'txt' in dobj:
            input_ids = self.tokenizer.encode(dobj['txt'])
            labels = input_ids.copy()
        else:
            messages = dobj['messages']
            for i in range(0, len(messages)):
                msg = messages[i]
                content = msg['content'].strip()
                if msg['role'] != 'assistant':
                    inp = self.tokenizer.encode(f'<|role|>{msg["role"]}<|says|>{content}<|end|>\n', add_special_tokens=False)
                    input_ids += inp
                    labels += [IGNORE_INDEX] * len(inp)
                else:
                    input_ids += self.role_assistant_says_ids
                    labels += [IGNORE_INDEX] * len(self.role_assistant_says_ids)
                    inp = self.tokenizer.encode(f'{content}<|end|>\n', add_special_tokens=False)
                    input_ids += inp
                    inp[-1] = IGNORE_INDEX 
                    labels += inp
        
        if NO_MASK:
            labels = input_ids.copy()

        assert len(input_ids) == len(labels) 
        
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        input_ids_tensor = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask_tensor = torch.zeros(self.max_length, dtype=torch.bool)
        labels_tensor = torch.full((self.max_length,), IGNORE_INDEX, dtype=torch.long)

        seq_length = min(len(input_ids), self.max_length)
        input_ids_tensor[:seq_length] = torch.tensor(input_ids[:seq_length], dtype=torch.long)
        attention_mask_tensor[:seq_length] = True
        labels_tensor[:seq_length] = torch.tensor(labels[:seq_length], dtype=torch.long)

        ret = {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'labels': labels_tensor,
        }

        self.cached_items[index] = ret
        return ret

