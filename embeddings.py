#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import spacy
from utils.utils import get_feature_embeddings

class Embeddings(nn.Module): 
    """
    Class that converts input words to their embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        """
        super(Embeddings, self).__init__()
        emb = get_feature_embeddings('gap-small.tsv')