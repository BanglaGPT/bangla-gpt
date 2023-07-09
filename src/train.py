#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COPYRIGHT NOTICE

    Copyright (c) 2023, Shamim Ahamed

This file is part of BangaGPT Project.

BangaGPT is free software: you can redistribute it and/or modify it under the 
terms of the  Attribution-NonCommercial-ShareAlike 4.0  International License

You should have received a copy of the License along with 'bangla-tokenizer' 
project, if not Please visit https://github.com/BanglaGPT/bangla-tokenizer.
  
"""


from tokenizers import ByteLevelBPETokenizer
from glob import glob
import os


vocab_size=52_000
min_frequency=5
special_tokens = [
    '[SOT]', '[PAD]', '[EOT]', '[UNK]', '[MASK]','[CLS]', '[SEP]',
]


files = glob('./Data/OSCAR_Processed/*')
print(files)

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=files, 
    vocab_size=vocab_size, 
    min_frequency=min_frequency, 
    special_tokens=special_tokens
)

if not os.path.exists('./out/tokenizer'):
    os.makedirs('./out/tokenizer')
tokenizer.save_model("./out/tokenizer", "banglagpt")



