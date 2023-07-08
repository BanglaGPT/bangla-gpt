#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 17:18:10 2023

@author: pavel
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
