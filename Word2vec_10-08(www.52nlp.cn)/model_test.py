# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 14:49:50 2016

@author: project
"""

import gensim
import time 

start_time1 = time.time()
model1 = gensim.models.Word2Vec.load_word2vec_format("wiki.en.text.vector", binary=False)
ans1 = model1.most_similar("man")
print "ans1", ans1 ,"time:" , (time.time()-start_time1)

start_time2 = time.time()
model2 = gensim.models.Word2Vec.load("wiki.en.text.model")
ans2 = model2.most_similar("man")
print "ans2", ans2 ,"time:" , (time.time()-start_time2)

"""
output @ Nvidia GTX960 2GDDR5
ans1 [(u'woman', 0.6980515718460083), (u'boy', 0.6065453290939331), (u'girl', 0.5980297327041626), (u'person', 0.5182406902313232), (u'stranger', 0.5158573985099792), (u'thug', 0.49287742376327515), (u'swordsman', 0.48199301958084106), (u'drunkard', 0.48089396953582764), (u'warrior', 0.4723621904850006), (u'incy', 0.4715915620326996)] time: 556.111608028
ans2 [(u'woman', 0.6980515718460083), (u'boy', 0.6065453886985779), (u'girl', 0.5980297327041626), (u'person', 0.5182406902313232), (u'stranger', 0.5158573389053345), (u'thug', 0.49287739396095276), (u'swordsman', 0.4819929003715515), (u'drunkard', 0.48089393973350525), (u'warrior', 0.47236213088035583), (u'incy', 0.47159141302108765)] time: 250.997016907
Using gpu device 0: GeForce GTX 960 (CNMeM is enabled with initial size: 80.0% of memory, cuDNN not available)
"""