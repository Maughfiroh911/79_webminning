#!/usr/bin/env python
# coding: utf-8

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
# 4. Cosine Similarity Measurement
# Cosine similarity digunakan untuk menghitung nilai kosinus sudut antara vektor dokumen dengan vektor kueri. Semakin kecil sudut yang dihasilkan, maka tingkat kemiripan esai semakin tinggi.
# Formula dari cosine similarity adalah sebagai berikut:
# 
# \begin{equation}
# \cos \alpha=\frac{\boldsymbol{A} \cdot \boldsymbol{B}}{|\boldsymbol{A}||\boldsymbol{B}|}=\frac{\sum_{i=1}^{n} \boldsymbol{A}_{i} X \boldsymbol{B}_{i}}{\sqrt{\sum_{i=1}^{n}\left(\boldsymbol{A}_{i}\right)^{2}} X \sqrt{\sum_{i=1}^{n}\left(\boldsymbol{B}_{i}\right)^{2}}}
# \end{equation}
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

# In[1]:


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
