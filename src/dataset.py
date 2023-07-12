#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COPYRIGHT NOTICE

    Copyright (c) 2023, Shamim Ahamed

This file is part of BangaGPT Project.

BangaGPT is free software: you can redistribute it and/or modify it under the 
terms of the  Attribution-NonCommercial-ShareAlike 4.0  International License

You should have received a copy of the License along with 'bangla-tokenizer' 
project, if not  Please visit https://github.com/BanglaGPT/bangla-gpt.

"""


from torch.utils.data import Dataset
from datasets import load_dataset

import json


if __name__ == '__main__':
    dataset = load_dataset(
        "oscar-corpus/OSCAR-2201",          
        use_auth_token=True,   
        language="bn",      
        streaming=True,  
        split="train"
    )
    
    
    for d in dataset:
        print(d['text']) # prints documents
        break
    
    print('ok')
