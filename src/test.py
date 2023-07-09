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
import pandas as pd


tokenizer = ByteLevelBPETokenizer(
    './out/tokenizer/banglagpt-vocab.json',
    './out/tokenizer/banglagpt-merges.txt'
    )




text = "কফি হাউসের সেই আড্ডাটা আজ আর নেই’ বলে বাঙালির যে চিরন্তন হা-হুতাশ, সে সম্পর্কে আমরা কমবেশি সবাই ওয়াকিফহাল।"
tokz_text = tokenizer.encode(text)

print(tokz_text.ids)


for i in tokz_text.ids:
    print(tokenizer.decode([i]))




d = []
for i in range(52000):
    token = tokenizer.decode([i])
    d.append({"Token ID": i, "Token": token})

pd.DataFrame(d).to_csv('./out/tokens.csv', index=False)









