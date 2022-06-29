#!/usr/bin/env python
# coding: utf-8

# ## Ekstraksi Fitur 

# Ekstraksi fitur merupakan proses untuk mencari nilai-nilai fitur yang terkandung dalam dokumen. Ekstraksi fitur menjadi bagian yang sangat penting dalam pengolahan dokumen pada mesin pencari karena sangat menentukan keberhasilan proses text mining. Salah satu metode ekstraksi fitur yang banyak digunakan dan populer adalah TF-IDF.

# ### TF-IDF 

# Term Frequency Inverse Document Frequency (TF-IDF) adalah algoritma pembobotan dokomen . 
# Term frequency (TF) adalah rasio dari jumlah kemunculan kata dalam dokumen.
# IDF adalah kemunculan kata terhadap keseluruhan dokumen dalam database.
# TF-IDF adalah ukuran yang dinormalisasi yang mempertimbangkan panjang dokumen.
# 
# $$
# \begin{aligned}
# &W=T F(I D F+1) \\
# &I D F=\log D / d f
# \end{aligned}
# $$

# Keterangan:
# 
# TF : Frekuensi kemunculan kata pada setiap dokumen
# 
# D: Jumlah dokumen
# 
# df: Jumlah dokumen yang mengandung sebuah term
# 
# IDF:Hasil log dari jumlah dokumen keseluruhan dibagi dengan jumlah dokumen yang mengandung sebuah term
# 
# W: Bobot frekuensi sebuah term pada sebuah dokumen

# In[1]:


import nltk
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.corpus import stopwords
#stop_words=set(nltk.corpus.stopwords.words('english', 'indonesian'))
stop_words = set(nltk.corpus.stopwords.words('indonesian'))

vect =TfidfVectorizer(stop_words=stop_words,max_features=1000) # to play with. min_df,max_df,max_features etc...


# In[10]:


#print(stop_words)


# In[11]:


vect_text=vect.fit_transform(df['abstrak_baru'])


# In[12]:


print(vect_text.shape)
print(vect_text)


# In[13]:


idf=vect.idf_
nilaiidf = [idf]


# In[14]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
#kata = [l]
#print(l)
print(l[0],l[-1])
print(dd)
#print(dd['metode'])


# In[15]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
kata = l
Nilai = idf
ax.plot(Nilai)
plt.show()

