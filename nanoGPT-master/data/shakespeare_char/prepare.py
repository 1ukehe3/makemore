"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import tiktoken
from tqdm import tqdm


# download the tiny shakespeare dataset
DATA_CACHE_DIR='data'
enc=tiktoken.get_encoding('gpt2')
encode=lambda s: enc.encode(s,allowed_special={'<|endoftext|>'})

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

def tokenize():
    eot=enc._special_tokens['<|endoftext|>']
    text=open(input_file_path, 'r',encoding='utf-8').read()
    text="<|endoftext|>"+text
    text=text.replace('\n\n','\n\n<|endoftext|>')

    tokens=encode(text)
    tokens_np=np.array(tokens,dtype=np.int32)
    
    print(f"length of dataset in characters: {len(text):,}")
  
# get all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(text)
    print("all the unique characters:", ''.join(chars))
   #S print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
#def encode(s):
#    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
#def decode(l):
#   return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
    n = len(tokens)
    train_tokens_np = tokens_np[:int(n*0.9)]
    val_tokens_np = tokens_np[int(n*0.9):]

# encode both to integers
#train_ids = encode(train_data)
#val_ids = encode(val_data)
#print(f"train has {len(train_ids):,} tokens")
#print(f"val has {len(val_ids):,} tokens")

# export to bin files
#train_ids = np.array(train_ids, dtype=np.uint16)
#val_ids = np.array(val_ids, dtype=np.uint16)

    train_filename=os.path.join(os.path.dirname(__file__), 'input_train.bin')
    val_filename=os.path.join(os.path.dirname(__file__), 'input_val.bin')
    print(train_tokens_np)
    print(val_tokens_np)
    with open(val_filename,'wb') as f:
        f.write(val_tokens_np.tobytes())
    with open(train_filename,'wb') as f:
        f.write(train_tokens_np.tobytes())
    print(f"保存{len(val_tokens_np)}tokens to {val_filename}")
    print(f"保存{len(train_tokens_np)}tokens to {train_filename}")

# save the meta information as well, to help us encode/decode later
    meta = {
      'vocab_size': vocab_size,
      'itos': itos,
      'stoi': stoi,
    }
    print(meta)
    with open(os.path.join(os.path.dirname(__file__), 'input_meta.pkl'), 'wb') as f:
         pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens

tokenize()