# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:12:37 2019

@author: Ananya Mukherjee
"""
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize 
eps = np.finfo(float).eps
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
#****************************************************************************************#
#Splits the dataset into traindata and testdata
def Split(data):
    len_data = len(data)
    len_train = int(len_data * 0.8)
    train = data.iloc[0:len_train]
    test = data.iloc[len_train:]
    return train,test

#Function to identify all the words that are noun in a given sentence
def getN(tagged):
    ls =[]
    for pair in tagged:
        if pair[1] == 'NN' or pair[1] == 'NNP' or pair[1] == 'NNS' or pair[1] == 'NNPS' or pair[1] == 'JJ':
            ls.append(pair[0])
    return ls

#Cleans the text by removing punctuations and replacing with spaces
def TextCleaner(text):   
    text=re.sub(r'(\d+)',r'',text)    
    text=text.replace(u'%','')   
    text=text.replace(u',','')
    text=text.replace(u'"','')
    text=text.replace(u'(','')
    text=text.replace(u')','')
    text=text.replace(u'"','')
    text=text.replace(u'“','')
    text=text.replace(u'”','')      
    text=text.replace(u':','')
    text=text.replace(u"'",'')
    text=text.replace(u"‘‘",'')
    text=text.replace(u"’’",'')
    text=text.replace(u"''",'')
    text=text.replace(u".",'')
    text=text.replace(u"-",'')
    text=text.replace(u"|",'')
    text=text.replace(u"!",'')
    text=text.replace(u"<br/>",'')
    text=text.replace(u"\n",'')
    return text

#Based upon the polarity, store the words into the feature attributes.
def appendAsPerPolarity(words,index,ls_words):
    if Dataset['Polarity'][index] == 'negative':
        ls_words[0].append(words)
    elif Dataset['Polarity'][index] == 'neutral':
        ls_words[1].append(words)
    elif Dataset['Polarity'][index] == 'positive':
        ls_words[2].append(words)
        
    return ls_words

#Convert into a single string.
def ConvertToSingleString(ls_2D):
    ls_1D_0 =  [s for S in ls_2D[0] for s in S] #Negative
    ls_1D_1 =  [s for S in ls_2D[1] for s in S] #Neutral
    ls_1D_2 =  [s for S in ls_2D[2] for s in S] #Positive
    
    #return [ls_1D_0,ls_1D_1,ls_1D_2]
    return [np.unique(ls_1D_0),np.unique(ls_1D_1),np.unique(ls_1D_2)]

def getProbability(CtgList):
    N = len(CtgList[0])
    Neu = len(CtgList[1])
    P = len(CtgList[2])
    tot = N + Neu + P
    
    
    NegProb = N/tot
    PosProb = P/tot
    NeuProb = Neu/tot
    
    return [NegProb,NeuProb,PosProb]
    
def WordMatchCount(TestVocab,CtgList):
    matchNeg = []
    matchNeu = []
    matchPos = []
    for ctg in CtgList:
        #check for the word match 
        matchNeg.append([item for item in TestVocab if item in ctg[0]])
        matchNeu.append([item for item in TestVocab if item in ctg[1]])
        matchPos.append([item for item in TestVocab if item in ctg[2]])
        
    return [matchNeg,matchNeu, matchPos]

def getLabel(matchlist,totalwords,labelProb,CtgProbability):
    predLabelProb = []
    for m in range(len(matchlist)):
        liklihood = 1
        for t in range(len(matchlist[m])):
            liklihood = liklihood * (len(matchlist[m][t])/totalwords) * CtgProbability[t][m]
        predLabelProb.append(liklihood*labelProb[m])
    return predLabelProb
            
  
#****************************************************************************************#
RawDataset = pd.read_csv('RestaurantDataset1.csv', encoding='utf-8')
#RawDataset = pd.read_csv('/Users/Sushom-Dell/Desktop/Ananya/New folder/CL2/PROJECT/Dataset.csv', encoding='utf-8')

#****************************************************************************************#
#Split Dataset 80% - 20%
Dataset, TestData = Split(RawDataset)
finalLabel = ['negative','neutral','positive']

#Create 2D array for each Category to hold the words list
service  = [[],[],[]]
food     = [[],[],[]]
ambience = [[],[],[]]
misc     = [[],[],[]]
price    = [[],[],[]]

CommentWiseWords = []
for i in range(len(Dataset)):
    CleanComment = TextCleaner(Dataset['Text'][i])
    #For each comment in the Dataset tokenize into words
    word_tokens = word_tokenize(CleanComment)
    tagged = nltk.pos_tag(word_tokens)
    words = getN(tagged)
    if Dataset['Category'][i] == 'service':
        service = appendAsPerPolarity(words,i,service)
    elif Dataset['Category'][i] == 'food':
        food = appendAsPerPolarity(words,i,food)
    elif Dataset['Category'][i] == 'price':
        price = appendAsPerPolarity(words,i,price)
    elif Dataset['Category'][i] == 'anecdotes/miscellaneous':
        misc = appendAsPerPolarity(words,i,misc)
    elif Dataset['Category'][i] == 'ambience':
        ambience = appendAsPerPolarity(words,i,ambience)

Categories = [service,food,ambience,misc,price] 
CtgProbability = []

#Flatten the 2D string into 1D
for i in range(len(Categories)):
    Categories[i] = ConvertToSingleString(Categories[i])
    #Calculate the probability for each Genre
    CtgProbability.append(getProbability(Categories[i]))
    
Pos = len(Dataset['Polarity'][Dataset['Polarity'] == 'positive'])
Neg = len(Dataset['Polarity'][Dataset['Polarity'] == 'negative'])
Neu = len(Dataset['Polarity'][Dataset['Polarity'] == 'neutral'])
total = Pos + Neg + Neu
posProbability = Pos/total
negProbability = Neg/total
neuProbability = Neu/total
labelProb = [negProbability,neuProbability,posProbability]
#********************TESTING***************************#
predictedLabels = []
match = []
for i in range(len(Dataset), len(RawDataset)):
    CleanComment = TextCleaner(TestData['Text'][i])
    #For each comment in the Dataset tokenize into words
    word_tokens = word_tokenize(CleanComment) 
    tagged = nltk.pos_tag(word_tokens) #POS tag the words
    words = getN(tagged) #Get Noun terms
    match = WordMatchCount(words,Categories) #Get the words that got matched
    predictions = getLabel(match,len(words),labelProb,CtgProbability)
    predictedLabels.append(finalLabel[np.argmax(predictions)])

testlist = TestData['Polarity'].tolist()

TestData['Label'][TestData['Label'] == 1] = 'positive'
TestData['Label'][TestData['Label'] == 0] = 'neutral'
TestData['Label'][TestData['Label'] == -1] = 'negative'
trueLabels = TestData['Label'].tolist()
cnt = 0
for i in range(len(predictedLabels)):
    if predictedLabels[i] == trueLabels[i]:
        cnt+=1

#*****************Classifier Measures*******************************************
accuracy = cnt/len(predictedLabels)
print('Accuracy:', accuracy)
print('****************** Confusion Matrix ********************** \n  Neg Neu Pos')
print(confusion_matrix(trueLabels, predictedLabels, labels=['negative', 'neutral','positive']))
print('*****************Classification Report *******************')
print(classification_report(trueLabels, predictedLabels))     
#*******************************************************************************

    

    
    
    
    
 
