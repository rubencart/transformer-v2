#!/usr/bin/env python
# -*- coding: utf-8 -*-
# date: 2018-12-02 20:54
import logging
import math

import spacy
import torch
import torch.nn as nn
from torchtext import data, datasets

from transformer.flow import make_model, run_epoch
from transformer.greedy import greedy_decode
from transformer.label_smoothing import LabelSmoothing
from transformer.multi_gpu_loss_compute import MultiGPULossCompute
from transformer.my_iterator import MyIterator, rebatch, batch_size_fn
from transformer.noam_opt import NoamOpt

# GPUs to use
# https://github.com/pytorch/pytorch/issues/11793
# https://github.com/pytorch/pytorch/issues/5587
devices = [0, 1, 2, 3]  # Or use [0, 1] etc for multiple GPUs

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                 eos_token=EOS_WORD, pad_token=BLANK_WORD)

logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S',
        level=logging.INFO,
)
logger = logging.getLogger(__name__)

device_objs = [torch.device('cuda:{}'.format(i) if torch.cuda.is_available() else 'cpu') for i in devices]
print(device_objs)

logger.info('loading spacy models')
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

MAX_LEN = 100
logger.info('loading dataset splits (can take some time, ~1 min)')
filter_long = lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN
train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TGT),
                                         filter_pred=filter_long)

# devs_x_bs = len(device_objs) * BATCH_SIZE
# dividable_size = len(train) // devs_x_bs * devs_x_bs
# train_div, train_rest = torch.utils.data.random_split(train, [dividable_size, len(train) - dividable_size])
# print('train: %s' % len(train))

logger.info('building vocabs')
MIN_FREQ = 2
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

logger.info('making iterators')
# Nb of tokens, not samples
BATCH_SIZE = 200
train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=device_objs[0], repeat=False,
                        sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)
valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=device_objs[0], repeat=False,
                        sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)

if True:
    logger.info('making big model')
    pad_idx = TGT.vocab.stoi[BLANK_WORD]
    model = make_model(len(SRC.vocab), len(TGT.vocab), num_layers=6, num_heads=8, d_model=128, d_ff=512)
    model.to(device_objs[0])
    model_par = nn.DataParallel(model, device_ids=devices)

    logger.info('making criterion & sending to gpu')
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.to(device_objs[0])

    logger.info('define optimizer')
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    total_src_tokens = sum([len(s.src) for s in train])
    total_trg_tokens = sum([len(s.trg) for s in train])
    num_batches_in_epoch = math.ceil(max(total_src_tokens, total_trg_tokens) / BATCH_SIZE)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('start training')
    logger.info('approx. %s batches per epoch' % num_batches_in_epoch)
    logger.info('number of parameters: {:,}'.format(num_trainable_params))

    for epoch in range(3):
        logger.info('****************************')
        logger.info('********* Epoch %s *********' % epoch)
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model_par,
                  MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt))

        model_par.eval()
        logger.info('running on validation set')
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par,
                         MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None))
        logger.info('validation loss: %s' % loss)
else:
    from transformer.batch import Batch
    from transformer.encoder_layer import EncoderLayer
    from transformer.encoder_decoder import EncoderDecoder
    from transformer.decoder import Decoder
    from transformer.decoder_layer import DecoderLayer
    from transformer.encoder import Encoder
    from transformer.encoder_layer import EncoderLayer
    from transformer.positionwise_feedforward import PositionwiseFeedForward
    from transformer.multiheaded_attention import MultiHeadedAttention
    from transformer.sublayer_connection import SublayerConnection
    from transformer.layer_norm import LayerNorm
    from transformer.embeddings import Embeddings
    from transformer.positional_encoding import PositionalEncoding
    from transformer.generator import Generator
    model = torch.load('iwslt.pt')

for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    print('Source:', end='\t')
    print(' '.join([SRC.vocab.itos[index] for index in src[0]]))
    print('')

    src_mask = (src != SRC.vocab.stoi[BLANK_WORD]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=TGT.vocab.stoi[BOS_WORD])
    print('Translation:', end='\t')
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == EOS_WORD:
            break
        print(sym, end=' ')
    print('')

    trg = batch.trg.transpose(0, 1)[0]
    print('Target:', end='\t')
    print(' '.join([TGT.vocab.itos[index] for index in trg]))
    print('')
    # break
