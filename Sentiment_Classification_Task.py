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


#Clean the input file
#need to lowercase,
#ValueError: I/O operation on closed file. - having issues cause file keeps closing

#pass clean input file
#split input file into each persons review and save order of reviews in dictionary
def open_posFile_and_extract_review():
    with open('test_pos_public.txt', encoding="utf8") as fp: 
        
        '''
        D ={}
        for num, review in enumerate(fp):
            #print("review {}: {}".format(num, review))
            D[num] = review
            for element in D[num]:
                element.readline()
                if element = ','|'.'|'-'|')'|'(':
        '''
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
        print(raw_word)              
                
        if raw_word in D:
            D[raw_word] += 1
        else:
            D[raw_word] = 1

    return D

def main():
    Reviews = open_posFile_and_extract_review()
    
    #for Reviews[1] in Reviews[0]:

        


if __name__ == '__main__':
    main()



