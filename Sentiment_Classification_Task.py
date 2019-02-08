#David Roster
#Programming Assignment #1

import sys  
import os
import string

print("You can do this :)")

#Task Description
#1. Extracting Features
    #Term Frequency-Inverse Document Frequency (TF-IDF)
    #Bag-of-Words (BoW)
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
        for review in reviews:
            #Pushes in a list of strings
            vector_list.append(vectorize(review))
    print(vector_list)
    return  vector_list

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

    return D

#gives us the frequency of the word in each document in the corpus.
# It is the ratio of number of times the word appears in a document
# compared to the total number of words in that document. 
def computeTF(wordDict, bow):
    #bow = nim of words in each review
    #worddict is passing vector_list in
    TFdict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        TFdict[word] = count/float(bowCount)
    return TFdict


def main():
    #returns vector list 
    Reviews = open_posFile_and_extract_review()
    
    

        


if __name__ == '__main__':
    main()



