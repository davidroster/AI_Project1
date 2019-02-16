#David Roster
#Programming Assignment #1

import sys  
import os
import string
import math
#import numpy

print("You can do this :)")

#Task Description
#1. Extracting Features
    #Term Frequency-Inverse Document Frequency (TF-IDF)
    #Bag-of-Words (BoW) <- So this is already done through Vectorize() and open_posFile_and_extract review
    #Optional Task: Removing Stop Words (Bonus 10 points)

#2. Implementing and Training Classifiers
    #Gaussian Naive Bayes classifier
    #Multinomial Naive Bayes classifier

#3. Writing a Report


#pass clean input file
#split input file into each persons review and save order of reviews in dictionary
def open_posFile_and_extract_review():
    with open('test_pos_public.txt', encoding="utf8") as fp:
        reviews = fp.readlines()
        vector_list = []

        review_sum = {}
        count_sum = 1

        for review in reviews:
            #Pushes in a list of strings
            D, individual_review_sum  = vectorize(review)
            vector_list.append(D)
            review_sum[count_sum] = individual_review_sum
            count_sum = count_sum + 1

    #Returns complete vector to main function
    return  vector_list, review_sum

def vectorize(individual_Review):
    #Takes in a list of strings
    word_list = individual_Review.strip().split()

    D = {}
    for word in word_list:
        #String processing bullshit
        raw_word = ""
        for char in word:
            if char.isalpha():
                raw_word += char.lower()
        #print(raw_word)              
                
        if raw_word in D:
            D[raw_word] += 1
        else:
            D[raw_word] = 1

    individual_Review_sum = sum(D.values())
    return D, individual_Review_sum


#gives us the frequency of the word in each document in the corpus.
# It is the ratio of number of times the word appears in a document
# compared to the total number of words in that document. 
def computeTF(total_reviews, count_word):
    #count_word = num of words in each review
    #total_reviews is passing vector_list in
    '''TF = frequency of word in doc/ total num of terms in doc '''
    
    TFVector = []
    incr_dict = {}
    count = 1

    #cycles through each dictionary in list
    for i in total_reviews:
        #cycles through each word in dictionary
        for word in i:
            indiv_dict_sum = count_word[count]
            word_count = i[word]
            TF = word_count / float(indiv_dict_sum)
            incr_dict[word] = TF
        #print(incr_dict)
        TFVector.append(incr_dict)
        count = count + 1

    return TFVector

def computeIDF(total_reviews, count_word):
    '''IDF = ln(total number of reviews / number of reviews with term in it)'''

    number_of_reviews = len(count_word)
    IDF_dict = {}

    for inc in total_reviews:
        for word in inc:
            if word in IDF_dict:
                IDF_dict[word] += 1
            else:
                IDF_dict[word] = 1
    
    for i in IDF_dict:
        IDF_dict[i] = math.log(number_of_reviews/float(IDF_dict[i]))

    return IDF_dict

def computeTF_IDF (TF, IDF):
    TF_IDF = []
    little_TFIDF = {}

    for i in TF:
        for word in i:
            little_TFIDF[word] = i[word] * IDF[word]
        TF_IDF.append(little_TFIDF)
        #print(little_TFIDF)

    return TF_IDF 

def main():
    #returns vector list -> Reviews
    #returns dictionary -> count_word
    Reviews, count_word = open_posFile_and_extract_review()
    #print(Reviews)
    #print(count_word)

    #print('\n')
    #print('\n')
    TF = computeTF(Reviews, count_word)
    IDF = computeIDF(Reviews, count_word)

    TF_IDF = computeTF_IDF(TF, IDF)

    #print(len(TF_IDF))

if __name__ == '__main__':
    main()



