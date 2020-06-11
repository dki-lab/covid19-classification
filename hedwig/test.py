# from transformers import ReformerModel, ReformerTokenizer
# import torch
#
# tokenizer = ReformerTokenizer.from_pretrained('google/reformer-crime-and-punishment')
# model =  ReformerModel.from_pretrained('google/reformer-crime-and-punishment')
#
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# print(input_ids.shape)
# outputs = model(input_ids)
#
# pooled_output = torch.mean(outputs[0], dim=1)
#
# last_hidden_states = outputs[0]


import torch
from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size
from transformers import RobertaTokenizer

config = LongformerConfig.from_pretrained('longformer-base-4096/')
# choose the attention mode 'n2', 'tvm' or 'sliding_chunks'
# 'n2': for regular n2 attantion
# 'tvm': a custom CUDA kernel implementation of our sliding window attention
# 'sliding_chunks': a PyTorch implementation of our sliding window attention
config.attention_mode = 'sliding_chunks'

model = Longformer.from_pretrained('longformer-base-4096/', config=config)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer.model_max_length = model.config.max_position_embeddings

SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
SAMPLE_TEXT = f'{tokenizer.cls_token}{SAMPLE_TEXT}{tokenizer.eos_token}'

input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

# TVM code doesn't work on CPU. Uncomment this if `config.attention_mode = 'tvm'`
# model = model.cuda(); input_ids = input_ids.cuda()

# Attention mask values -- 0: no attention, 1: local attention, 2: global attention
attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
attention_mask[:, [1, 4, 21,]] =  2  # Set global attention based on the task. For example,
                                     # classification: the <s> token
                                     # QA: question tokens

# padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
input_ids, attention_mask = pad_to_window_size(
        input_ids, attention_mask, config.attention_window[0], tokenizer.pad_token_id)

output = model(input_ids, attention_mask=attention_mask)[0]

