#David Roster
#Programming Assignment #1

import sys  
import os
import string

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
def computeTF(wordDict, bow):
    #bow = num of words in each review
    #worddict is passing vector_list in
    '''TF = frequency of word in doc/ total num of terms in doc '''
    print('\n')
    print('\n')
    print("Enter ComputerTF Function\n")
    print("WordDict type is ..." + str(wordDict))
    print("bow type is ..." + str(bow))
            #for word_incrementor in indiviudal_dictionary:
        #TypeError: 'NoneType' object is not iterable
        #trying to use count var instead
    
    TFVector = []
    count = 0
    '''
    #Iterate through vector_list that we passed in
    for total_count in wordDict:
        #pops the first dictionary
        indiviudal_dictionary = wordDict.remove(total_count)
        for word_incrementor in indiviudal_dictionary:
            individual_TFdict = {}
            word_frequency = indiviudal_dictionary[word_incrementor]
            print("word frequency is ... " + word_frequency)
            individual_TFdict[word_incrementor] = word_frequency/float(bow[total_count])
            print("individual_TFdict is ... " + individual_TFdict)
        TFVector.append(individual_TFdict)
        count = count + 1

    
#        individual_TFdict = {}
#        #bowCount = len(bow)
#        for word in wordDict:
#            individual_TFdict[word] = word/float(bow[word])
#            print(individual_TFdict[word])
#        return individual_TFdict
    '''
    return TFVector

def main():
    #returns vector list -> Reviews
    #returns dictionary -> count_word
    Reviews, count_word = open_posFile_and_extract_review()
    print(Reviews)
    print(count_word)

    print('\n')
    print('\n')

    TF = computeTF(Reviews, count_word)
    print(TF)

if __name__ == '__main__':
    main()



