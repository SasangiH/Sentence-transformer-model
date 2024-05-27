#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install sentence_transformers')


# In[2]:


#Import required libraries
import pandas as pd
import numpy as np
import spacy
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
nltk.download('punkt')

import sentence_transformers
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')


# In[ ]:




