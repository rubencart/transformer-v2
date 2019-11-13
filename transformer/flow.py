# -*- coding: utf-8 -*-
# date: 2018-12-02 11:35
import copy
import logging
import time

import torch.nn as nn

from .decoder import Decoder
from .decoder_layer import DecoderLayer
from .embeddings import Embeddings
from .encoder import Encoder
from .encoder_decoder import EncoderDecoder
from .encoder_layer import EncoderLayer
from .generator import Generator
from .multiheaded_attention import MultiHeadedAttention
from .positionwise_feedforward import PositionwiseFeedForward
from .positional_encoding import PositionalEncoding

logger = logging.getLogger(__name__)


def make_model(src_vocab, tgt_vocab, num_layers=6, d_model=512, d_ff=2048, num_heads=8, dropout=0.1):
    """
    Helper: Construct a model from hyperparameters.
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(num_heads, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), num_layers),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), num_layers),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def run_epoch(data_iter, model, loss_compute):
    """
    Standard Training and Logging Function
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            logging.info('Epoch step: %d Loss %f Tokens per Sec: %f' % (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens
