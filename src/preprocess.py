#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COPYRIGHT NOTICE

    Copyright (c) 2023, Shamim Ahamed

This file is part of BangaGPT Project.

BangaGPT is free software: you can redistribute it and/or modify it under the 
terms of the  Attribution-NonCommercial-ShareAlike 4.0  International License

You should have received a copy of the License along with 'bangla-tokenizer' 
project, if not  Please visit https://github.com/BanglaGPT/bangla-tokenizer.

"""


from datasets import load_dataset
import os
from tqdm.auto import tqdm

dataset = load_dataset("oscar-corpus/OSCAR-2201",
                        use_auth_token=True,
                        language="bn", 
                        streaming=True, 
                        split="train+valid+test")


if not os.path.exists('./Data/OSCAR_Processed'):
    os.mkdir('./Data/OSCAR_Processed')


N = 10000
count = 0
text = ''
for idx, d in tqdm(enumerate(dataset)):
    text += '\n' + d['text']
    
    if idx % N == 0 and idx!= 0:
        with open(f'./Data/OSCAR_Processed/chunk_{count}.txt', 'w') as f:
            f.write(text)
        text = ''
        count += 1
