#!/usr/bin/env python
# coding: utf-8

# ## Preprocessing

# ### Menggunakan 50 Data PTA, 25 Kategori RPL dan 25 Kategori CAI 

# #### Mengimport Library yang dibutuhkan 

# In[1]:


import numpy as np
import pandas as pd

import nltk
nltk.download('popular')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 


# ### Menampilkan dokumen  dari hasil Crawling

# In[2]:


df = pd.read_csv("abstrakv1.csv")
df.head()


# ## PREPROCESSING 

# Preprocessing merupakan proses awal yang akan mentransformasikan data masukan menjadi data dengan format yang sesuai dan siap untuk diproses (Setyohadi et al., 2017).  Tahapan dalam proses text preprocessing antara lain yaitu cleansing, tokenizing, filtering, stopword dan lain-lain. Penerapan tahap text preprocessing dapat berbeda-beda untuk setiap bahasa tergantung dengan kebutuhan, karena perbedaan dalam bahasa tentu memiliki arti yang berbeda untuk tiap katanya.

# ### Case Folding
# Case folding merupakan langkah dalam pengolahan data yang bertujuan untuk mengubah ataumenghilangkan semua huruf kapital pada dokumen menjadi huruf kecil

# In[3]:


data_abstrak = df['abstrak']
data_abstrak = data_abstrak.str.lower()
df['abstrak_baru'] = data_abstrak
df['abstrak_baru']


# ### Menghilangkan Angka dan Remove Punctuation 
# Remove Punctuation merupakan teknik penghilangan tanda baca yang digunakan dalam sebuah teks untuk membedakan antara kalimat dan bagian penyusunnya dan untukmemperjelas maknanya

# In[4]:


import re #library re (regular expression (regex)) dapat digunakan untuk menghapus karakter angka.
import string #library String digunakan untuk memproses string Python standar

def remove_number(text):
    return  re.sub(r"\d+", "", text)

df['abstrak_baru'] = df['abstrak_baru'].apply(remove_number)

#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

df['abstrak_baru'] = df['abstrak_baru'].apply(remove_punctuation)


# In[5]:


df['abstrak_baru']


# ### Stopword 

# Stopword adalah stopwords merupakan kata yang diabaikan dalam pemrosesan dan biasanya disimpan di dalam stop lists. Stop list ini berisi daftar kata umum yang mempunyai fungsi tapi tidak mempunyai arti. Contoh stopword dalam bahasa Indonesia adalah “yang”, “dan”, “di”, “dari”, dll. 

# In[6]:


def stopping_word(contents):    
    data_kata = []
    stop_words = stopwords.words('english')
    stop_words2 = stopwords.words('indonesian')
    stop_words.extend(stop_words2)
    jmlData = contents.shape 
    for i in range(jmlData[0]):
        word_tokens = word_tokenize(contents[i])
        # print(word_tokens)
            
        word_tokens_no_stopwords = [w for w in word_tokens if not w in stop_words]

        special_char = "+=`@_!#$%^&*()<>?/\|}{~:;.[],1234567890‘’'" + '"“”●'
        out_list = [''.join(x for x in string if not x in special_char) for string in word_tokens_no_stopwords]
        # print('List after removal of special characters:', out_list)

        while '' in out_list:
            out_list.remove('')
        data_kata.append(out_list)
    return data_kata


# In[7]:


stop_kata = stopping_word(df['abstrak_baru'])
df['stop_kata'] = stop_kata


# In[8]:


df['stop_kata']

