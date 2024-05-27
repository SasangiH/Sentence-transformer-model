#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', '"./utils.ipynb"')


# In[2]:


def vector_embeddings(preprocessed_sentence):
  sentence_vectors = {}

  #Get Sentence Transformer
  model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

  for sentence in preprocessed_sentence:
    #Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentence)
    sentence_vectors[sentence] = embeddings

  # #return embeddings

  # for embedding, sentences in zip(embeddings, preprocessed_sentence):
  #   sentence_vectors[sentences] = embeddings
  return sentence_vectors


# In[ ]:




