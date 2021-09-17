#!/usr/bin/env python
# coding: utf-8

# In[1]:


from konlpy.tag import Kkma
import os
java_home=os.environ['JAVA_HOME']

kkma=Kkma()

kkma.sentences('한국어 분석을 시작합니다 재미있어요~~')


# In[2]:


kkma.nouns('한국어 분석을 시작합니다 재미있어요~~')


# In[3]:


kkma.pos('한국어 분석을 시작합니다 재미있어요~~')


# In[4]:


from konlpy.tag import Hannanum
hannanum = Hannanum()

hannanum.nouns('한국어 분석을 시작합니다 재미있어요~~')


# In[5]:


hannanum.morphs('한국어 분석을 시작합니다 재미있어요~~')


# In[6]:


hannanum.pos('한국어 분석을 시작합니다 재미있어요~~')


# In[7]:


from konlpy.tag import Twitter
t = Twitter()


# In[8]:


t.nouns('한국어 분석을 시작합니다 재미있어요~~')


# In[9]:


t.morphs('한국어 분석을 시작합니다 재미있어요~~')


# In[17]:


t.pos('한국어 분석을 시작합니다 재미있어요~~')


# In[21]:


t.pos('나는 단팥죽을 좋아합니다')


# In[23]:


from wordcloud import WordCloud, STOPWORDS

import numpy as np
from PIL import Image

import os

print(os.getcwd())

text=open('09. alice.txt').read()
alice_mask=np.array(Image.open('09. alice_mask.png'))

stopwords = set(STOPWORDS)
stopwords.add("said")


# In[45]:


import matplotlib.pyplot as plt
import platform

path = "C:/Windows/Fonts/Corbel.ttf"
from matplotlib import font_manager, rc
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~')
    
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


plt.figure(figsize=(8,8))
plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[27]:


wc = WordCloud(background_color='white', max_words=2000, mask=alice_mask, stopwords=stopwords)
wc = wc.generate(text)
wc.words_


# In[28]:


plt.figure(figsize=(12,12))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[29]:


text = open('09. a_new_hope.txt').read()

text = text.replace('HAN', 'Han')
text = text.replace("LUKE'S", 'Luke')

mask = np.array(Image.open('09. stormtrooper_mask.png'))


# In[30]:


stopwords = set(STOPWORDS)
stopwords.add("int")
stopwords.add("ext")


# In[31]:


wc = WordCloud(max_words=1000, mask=mask, stopwords=stopwords, margin=10, random_state=1).generate(text)

default_colors = wc.to_array()


# In[32]:


import random
def gray_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'hsl(0, 0%%, %d%%)' % random.randint(60, 100)


# In[33]:


plt.figure(figsize=(12,12))
plt.imshow(wc.recolor(color_func=gray_color_func, random_state=3),
          interpolation='bilinear')
plt.axis('off')
plt.show()


# In[34]:


import nltk


# In[36]:


from konlpy.corpus import kobill
files_ko = kobill.fileids()
doc_ko = kobill.open('1809890.txt').read()


# In[37]:


doc_ko


# In[38]:


from konlpy.tag import Twitter; t=Twitter()
tokens_ko = t.nouns(doc_ko)
tokens_ko


# In[39]:


ko = nltk.Text(tokens_ko, name='대한민국 국회 의안 제 189890호')


# In[40]:


print(len(ko.tokens)) # 토큰의 개수 확인
print(len(set(ko.tokens))) # unique 토큰의 개수 확인
ko.vocab() # Frequency Distribution 확인


# In[41]:


print(ko.count('육아휴직')) # 영어 찾을때는 앞에 en?


# In[42]:


ko.concordance('육아휴직', lines=5) # '육아휴직' 단어가 들어있는 문장 5개 출력


# In[55]:


import matplotlib.font_manager as fm
import matplotlib as mpl

mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12,6))
ko.plot(50)

path = 'C:/Windows/Fonts/HMKMRHD.TTF'
font_name = fm.FontProperties(fname=path, size=50).get_name()
plt.rc('font', family=font_name)

plt.show()


# In[56]:


stop_words = ['.', '(', ')', ',', "'", '%', '-', 'X', ').', '×','의','자','에','안','번',
                      '호','을','이','다','만','로','가','를']
ko = [each_word for each_word in ko if each_word not in stop_words]
ko


# In[57]:


ko = nltk.Text(ko, name='대한민국 국회 의안 제 1809890호')
plt.figure(figsize=(12,6))
ko.plot(50)
plt.show()


# In[58]:


ko.count('초등학교')


# In[59]:


plt.figure(figsize=(12,6))
ko.dispersion_plot(['육아휴직', '초등학교', '공무원'])


# In[60]:


ko.concordance('초등학교')


# In[67]:


import nltk
nltk.download('stopwords')

ko.collocations()


# In[68]:


ko.collocations()


# In[71]:


data = ko.vocab().most_common(150)

# font_path=''C:/Windows/Fonts/HMKMRHD.TTF''

wordcloud = WordCloud(font_path='C:/Windows/Fonts/HMKMRHD.TTF',
                     relative_scaling = 0.2,
                     background_color = 'white',
                     ).generate_from_frequencies(dict(data))
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[72]:


from nltk.tokenize import word_tokenize
import nltk 
# 자연어 처리 및 분석, 텍스트마이닝을 위한 파이썬 패키지
# 토큰생성, 형태소 분석, 품사태깅 등의 다양한 기능을 제공하고 예제로 활용할 수 있는 말뭉치도 제공


# In[73]:


train = [('i like you', 'pos'),
        ('i hate you', 'neg'),
        ('you like me', 'neg'),
        ('i like her', 'pos')]


# In[75]:


import nltk
nltk.download('punkt')


# In[76]:


all_words = set(word.lower() for sentence in train
               for word in word_tokenize(sentence[0]))
all_words


# In[77]:


t = [({word : (word in word_tokenize(x[0])) for word in all_words}, x[1])
    for x in train]
t


# In[78]:


classifier = nltk.NaiveBayesClassifier.train(t)
classifier.show_most_informative_features()


# In[79]:


test_sentence = 'i like MeRui'
test_sent_features = {word.lower(): 
                        (word in word_tokenize(test_sentence.lower()))
                           for word in all_words}
test_sent_features


# In[80]:


classifier.classify(test_sent_features)


# In[81]:


from konlpy.tag import Twitter


# In[82]:


pos_tagger = Twitter()


# In[83]:


train = [('메리가 좋아', 'pos'),
        ('고양이도 좋아', 'pos'),
        ('난 수업이 지루해', 'neg'),
        ('메리는 이쁜 고양이야', 'pos'),
        ('난 마치고 메리랑 놀거야', 'pos')]


# In[84]:


all_words = set(word.lower() for sentence in train
               for word in word_tokenize(sentence[0]))
all_words


# In[85]:


t = [({word : (word in word_tokenize(x[0])) for word in all_words}, x[1])
    for x in train]
t


# In[86]:


classifier = nltk.NaiveBayesClassifier.train(t)
classifier.show_most_informative_features()


# In[87]:


test_sentence = '난 수업이 마치면 메리랑 놀거야'


# In[88]:


test_sent_features = {word.lower():
                     (word in word_tokenize(test_sentence.lower()))
                     for word in all_words}
test_sent_features


# In[89]:


classifier.classify(test_sent_features)


# In[93]:


def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]


# In[94]:


train_docs = [(tokenize(row[0]), row[1]) for row in train]
train_docs


# In[95]:


tokens = [t for d in train_docs for t in d[0]]
tokens


# In[96]:


def term_exists(doc):
    return {word: (word in set(doc)) for word in tokens}


# In[97]:


train_xy = [(term_exists(d),c) for d,c in train_docs]
train_xy


# In[98]:


classifier = nltk.NaiveBayesClassifier.train(train_xy)


# In[99]:


test_sentence=[("난 수업이 마치면 메리랑 놀거야")]


# In[100]:


test_docs=pos_tagger.pos(test_sentence[0])
test_docs


# In[101]:


classifier.show_most_informative_features()


# In[102]:


test_sent_features={word: (word in tokens) for word in test_docs}
test_sent_features


# In[103]:


classifier.classify(test_sent_features)


# In[107]:


from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import platform

text = open('protect_dolphins.txt', encoding='UTF8').read()
text = text.replace('HAN', 'Han')
text = text.replace("LUKE'S", 'Luke')

mask = np.array(Image.open('protect_dolphins.png'))
stopwords = set(STOPWORDS)
stopwords.add("int")
stopwords.add("ext")

wc = WordCloud(max_words=1000, mask=mask, stopwords=stopwords,
               margin=10, random_state=1).generate(text)
              # margin은 여백?
              # if random object is given, this ti used for generating random numbers
default_colors = wc.to_array()    # convert to array for recoloring

import random
def grey_color_func(word, font_size, position, orientation,
                    random_state = None, **kwargs):
    return 'hsl(200, 100%%, %d%%)' % random.randint(60, 100)
          # hsl(색상, 채도, 명도)
          # 색상 0=red, 120=green, 240=blue. 0~360까지 가능
          # 채도 0%는 회색, 100%는 풀걸러
          # 명도 0%는 블랙, 100%는 화이트

plt.figure(figsize=(12, 12))
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
           interpolation='bilinear')
plt.axis('off')
plt.show()


# In[110]:


from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import platform

text = open('protect_dolphins.txt', encoding='UTF8').read()
text = text.replace('HAN', 'Han')
text = text.replace("LUKE'S", 'Luke')

mask = np.array(Image.open('flower.png'))
stopwords = set(STOPWORDS)
stopwords.add("int")
stopwords.add("ext")

wc = WordCloud(max_words=1000, mask=mask, stopwords=stopwords,
               margin=10, random_state=1).generate(text)
              # margin은 여백?
              # if random object is given, this ti used for generating random numbers
default_colors = wc.to_array()    # convert to array for recoloring

import random
def grey_color_func(word, font_size, position, orientation,
                    random_state = None, **kwargs):
    return 'hsl(340, 100%%, %d%%)' % random.randint(60, 100)
          # hsl(색상, 채도, 명도)
          # 색상 0=red, 120=green, 240=blue. 0~360까지 가능
          # 채도 0%는 회색, 100%는 풀걸러
          # 명도 0%는 블랙, 100%는 화이트

plt.figure(figsize=(12, 12))
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
           interpolation='bilinear')
plt.axis('off')
plt.show()


# In[111]:


from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import platform

text = open('protect_dolphins.txt', encoding='UTF8').read()
text = text.replace('HAN', 'Han')
text = text.replace("LUKE'S", 'Luke')

mask = np.array(Image.open('apple.png'))
stopwords = set(STOPWORDS)
stopwords.add("int")
stopwords.add("ext")

wc = WordCloud(max_words=1000, mask=mask, stopwords=stopwords,
               margin=10, random_state=1).generate(text)
              # margin은 여백?
              # if random object is given, this ti used for generating random numbers
default_colors = wc.to_array()    # convert to array for recoloring

import random
def grey_color_func(word, font_size, position, orientation,
                    random_state = None, **kwargs):
    return 'hsl(0, 100%%, %d%%)' % random.randint(60, 100)
          # hsl(색상, 채도, 명도)
          # 색상 0=red, 120=green, 240=blue. 0~360까지 가능
          # 채도 0%는 회색, 100%는 풀걸러
          # 명도 0%는 블랙, 100%는 화이트

plt.figure(figsize=(12, 12))
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
           interpolation='bilinear')
plt.axis('off')
plt.show()

