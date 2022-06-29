#!/usr/bin/env python
# coding: utf-8

# ## Maughfirotu Jannah

# ## CRAWLING DATA 

# Data adalah  kumpulan atau catatan fakta yang menggambarkan suatu kejadian, dan masih dalam bentuk data mentah sehingga perlu diolah lebih lanjut untuk menghasilkan informasi. Crawling adalah proses menjelajahi web dan mengunduh halaman web secara otomatis untuk mengumpulkan informasi. Program yang khusus bertugas melakukan crawling disebut Crawler. (Hanifah & Nurhasanah, 2018)

# In[1]:


import scrapy

class Crawling(scrapy.Spider):
    name = "Crawling"
    
    def start_requests(self):
        x = 100000
        for i in range (1,5):
            x +=1
            urls = [
                'https://pta.trunojoyo.ac.id/welcome/detail/040411'+str(x),
                'https://pta.trunojoyo.ac.id/welcome/detail/050411'+str(x),
                'https://pta.trunojoyo.ac.id/welcome/detail/060411'+str(x),
                'https://pta.trunojoyo.ac.id/welcome/detail/070411'+str(x),
                'https://pta.trunojoyo.ac.id/welcome/detail/080411'+str(x),
                'https://pta.trunojoyo.ac.id/welcome/detail/090411'+str(x),
            ]
            for url in urls:
                yield scrapy.Request(url=url, callback=self.parse)
    
    def parse(self, response):
        yield{
            'judul' : response.css('#content_journal > ul > li > div:nth-child(2) > a').extract(),
            'abstrak' : response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) > p').extract()
        }


# In[2]:


scrapy crawl Crawling


# ### Menggunakan 50 Data PTA, 25 Kategori RPL dan 25 Kategori CAI 

# In[2]:


#library numpy berfungsi untuk melakukan operasi vektor dan matriks dengan mengolah array dan array multidimensi. 
#Library NumPy digunakan untuk kebutuhan dalam menganalisis data.
get_ipython().system('pip install numpy')


# In[3]:


#library pandas digunakan untuk mengelola data berbentuk tabel atau disebut dengan dataframe
#Library yang digunakan untuk Penanganan, manipulasi, dan analisis data
get_ipython().system('pip install pandas')


# In[4]:


#NLTK adalah singkatan dari Natural Language Tool Kit
#library yang digunakan untuk membantu kita dalam bekerja dengan teks
#Library ini memudahkan kita untuk memproses teks seperti melakukan classification, tokenization, stemming, tagging, parsing, dan semantic reasoning.
get_ipython().system('pip install nltk')


# In[5]:


#Library sastrawi digunakan untuk melakukan tokenisasi (tokenize) dan membuang stopwords dari teks.
get_ipython().system('pip install Sastrawi')


# In[1]:


import numpy as np
import pandas as pd

import nltk
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

# In[9]:


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


# ## Topic Modelling LSA 

# Latent semantic analysis (LSA) adalah teknik dalam pemrosesan bahasa alami , khususnya semantik distribusional , menganalisis hubungan antara satu set dokumen dan istilah yang dikandungnya dengan menghasilkan satu set konsep yang terkait dengan dokumen dan istilah. 
# langkah-langkah LSA sebagai berikut:
# 1. Teks Preprocessing
# 2. Term-document Matrix
# 3. Singular Value Decomposition
# Singular Value Decomposition (SVD) adalah salah satu teknik reduksi dimensi yang bermanfaat untuk memperkecil nilai kompleksitas dalam pemrosesan term-document matrix. SVD merupakan teorema aljabar linier yang menyebutkan bahwa persegi panjang dari term-document matrix dapat dipecah/didekomposisikan menjadi tiga matriks, yaitu :
# – Matriks ortogonal U
# – Matriks diagonal S
# – Transpose dari matriks ortogonal V
# Yang dirumuskan dengan :
# 
# $A_{m n}=U_{m m} x S_{m n} x V_{n n}^{T}$
# 
# Keterangan:
# 
# $$\begin{array}{ll}\mathrm{A}_{\mathrm{mn}}= & \text { matriks awal } \\ \mathrm{U}_{\mathrm{mm}}= & \text { matriks ortogonal U } \\ \mathrm{S}_{\mathrm{mn}}= & \text { matriks diagonal } \mathrm{s} \\ \mathrm{V}_{n \pi}^{\top}= & \text { transpose matriks ortogonal } \mathrm{V}\end{array}$$
# 
# Hasil dari proses SVD adalah vektor yang akan digunakan untuk menghitung similaritasnya dengan pendekatan cosine similarity.
# 
# 
# 4.Cosine Similarity Measurement
# Cosine similarity digunakan untuk menghitung nilai kosinus sudut antara vektor dokumen dengan vektor kueri. Semakin kecil sudut yang dihasilkan, maka tingkat kemiripan esai semakin tinggi.
# Formula dari cosine similarity adalah sebagai berikut:
# 
# $\begin{equation}
# \cos \alpha=\frac{\boldsymbol{A} \cdot \boldsymbol{B}}{|\boldsymbol{A}||\boldsymbol{B}|}=\frac{\sum_{i=1}^{n} \boldsymbol{A}_{i} X \boldsymbol{B}_{i}}{\sqrt{\sum_{i=1}^{n}\left(\boldsymbol{A}_{i}\right)^{2}} X \sqrt{\sum_{i=1}^{n}\left(\boldsymbol{B}_{i}\right)^{2}}}
# \end{equation}$
#  
# Keterangan:
# 
# A $\quad=$ vektor dokumen
# 
# B $\quad=$ vektor kueri
# 
# A $\cdot \mathbf{B}=$ perkalian $\operatorname{dot}$ vektor $\mathrm{A}$ dan vektor $\mathrm{B}$
# 
# $|\mathrm{A}| \quad=$ panjang vektor $\mathrm{A}$
# 
# $\left.\right|^{\boldsymbol{B}} \mid \quad=$ panjang vektor $B$
# 
# $\mid \boldsymbol{A}_{|| \mathrm{B} \mid}=$ cross product antara $|\mathrm{A}|$ dan $|\mathrm{B}|$
# 
# 
# Dari hasil cosine similarity, akan didapatkan nilai yang akan dibandingkan dengan penilaian manusia untuk diuji selisih nilainya.

# In[16]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=50, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[17]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[18]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# In[19]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,3,3])
Nilai = topic
topic2 = l
ax.plot(topic2)
plt.show()


# In[20]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# In[21]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# ### Daftar Pustaka
# - Nawassyarif, Julkarnain M, K. R. A. (2020). 338108-sistem-informasi-pengolahan-data-ternak-30b9d1b3 ( Pengertian Sistem infomasi. Jurnal JINTEKS, 2(1), 32–39.
# - Hanifah, R., & Nurhasanah, I. S. (2018). Implementasi Web Crawling Untuk Mengumpulkan Web Crawling Implementation for Collecting. Jurnal Teknologi Informasi Dan Ilmu Komputer (JTIIK), 5(5), 531–536. https://doi.org/10.25126/jtiik20185842
# - https://docs.scrapy.org/en/latest/intro/tutorial.html
# - Setyohadi, D. B., Kristiawan, F. A., & Ernawati, E. (2017). Perbaikan Performansi Klasifikasi Dengan Preprocessing Iterative Partitioning Filter Algorithm. Telematika, 14(01), 12–20. https://doi.org/10.31315/telematika.v14i01.1960
# - Merinda Lestandy, Abdurrahim Abdurrahim, & Lailis Syafa’ah. (2021). Analisis Sentimen Tweet Vaksin COVID-19 Menggunakan Recurrent Neural Network dan Naïve Bayes. Jurnal RESTI (Rekayasa Sistem Dan Teknologi Informasi), 5(4), 802–808. https://doi.org/10.29207/resti.v5i4.3308
# - Prihatini, P. M. (2016). Implementasi Ekstraksi Fitur Pada Pengolahan Dokumen Berbahasa Indonesia. Jurnal Matrix, 6(3), 174–178.
# - https://socs.binus.ac.id/2015/08/03/penggunaan-latent-semantic-analysis-lsa-dalam-pemrosesan-teks/
# - https://www.kaggle.com/code/rajmehra03/topic-modelling-using-lda-and-lsa-in-sklearn/notebook
