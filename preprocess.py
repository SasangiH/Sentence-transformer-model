#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', '"./utils.ipynb"')


# In[2]:


def preprocess_sentences(sentences):
  # Tokenization, lower casing and Vocabulary Creation
  vocabulary = set()
  preprocessed_sentence = []
  for sentence in sentences:
      tokens = word_tokenize(sentence)

      # Lowercase all words
      tokens = [tokens.lower() for tokens in tokens]

      # Join the words back into a single string
      preprocessed_text = ' '.join(tokens)
      preprocessed_sentence.append(preprocessed_text)

      vocabulary.update(tokens)

  # Assign unique indices to words in the vocabulary
  word_to_index = {word: i for i, word in enumerate(vocabulary)}

  return vocabulary, word_to_index, preprocessed_sentence

def preprocess_query(query_sentence):
  # Tokenization, lower casing and Vocabulary Creation
  #preprocessed_sentence = []
  tokens = word_tokenize(query_sentence)

  # Lowercase all words
  tokens = [tokens.lower() for tokens in tokens]

  # # Join the words back into a single string
  # preprocessed_text = ' '.join(tokens)
  # preprocessed_sentence.append(preprocessed_text)

  return tokens


# In[ ]:




