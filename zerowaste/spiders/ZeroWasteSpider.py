# -*- coding: utf-8 -*-
"""
Author: Brandi Beals
Description: Assignment 1 Spider
"""

import scrapy
import os.path
import string
import re
from zerowaste.items import ZerowasteItem

class ZeroWasteSpider(scrapy.Spider):
    name = "zerowaste"

    def start_requests(self):
        urls = [
            'https://en.wikipedia.org/wiki/Zero_waste',
            'https://mashable.com/article/zero-waste/',
            'https://www.glasspantrymilwaukee.com/zerowaste/',
            'https://trashisfortossers.com/compost-lets-break-it-down-literally/',
            'https://www.goingzerowaste.com/blog-posts-for-beginners',
            'https://www.reddit.com/r/ZeroWaste/',
            'https://zerowastehome.com/blog/',
            'https://zerowaste.com/'
        ]
        
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        re_ws = re.compile(r"\s+", re.MULTILINE)
        #code = 
        url = response.url
        titleraw = response.css('title::text').get()
        title = re_punc.sub('', titleraw)
        title = re_ws.sub('', title)
        bodyraw = response.body
        body = response.css('p::text').getall() 
        body = ''.join(body)
        
        # store html in directory
        page_dirname = 'html'
        filename = '%s.html' % title
        with open(os.path.join(page_dirname,filename), 'wb') as f:
            f.write(bodyraw)
        self.log('Saved file %s' % filename)
        
        # follow links
        linksraw = response.css('a::attr(href)').getall()
        links = ','.join(linksraw)
        #yield from response.follow_all(css='li.next a::attr(href)', callback=self.parse)
        for link in response.css('a::attr(href)'):
            url = response.urljoin(link.extract())
            yield scrapy.Request(url, callback=self.parse_link)
        
        # return item details
        item = ZerowasteItem()
        item['url'] = url
        item['title'] = title
        item['body'] = body
        item['links'] = links
        return item

    def parse_link(self, response):
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        re_ws = re.compile(r"\s+", re.MULTILINE)
        code = response.xpath('//td[@id="item_id"]/text()').re(r'ID: (\d+)')
        url = response.url
        titleraw = response.css('title::text').get()
        title = re_punc.sub('', titleraw)
        title = re_ws.sub('', title)
        bodyraw = response.body
        body = response.css('p::text').getall() 
        body = ''.join(body)
        
        # store html in directory
        page_dirname = 'html'
        filename = '%s.html' % title
        with open(os.path.join(page_dirname,filename), 'wb') as f:
            f.write(bodyraw)
        self.log('Saved file %s' % filename)

        # return item details
        item = ZerowasteItem()
        item['code'] = code
        item['url'] = url
        item['title'] = title
        item['body'] = body
        return item

