#!/usr/bin/env python
# coding: utf-8

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


# Digunakan untuk mengcrawl data dan hasil crawlnya dalam bentuk csv

# In[2]:


scrapy crawl Crawling -o data.csv

