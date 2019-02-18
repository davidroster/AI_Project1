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
def open_File_train_and_extract_review(filename):
    #with open('test_pos_public.txt', encoding="utf8") as fp:
    with open(filename, encoding="utf8") as fp:
        temp = ""
        for line in fp:
            temp = temp + line
        reviews = temp.split("<br /><br />")
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
    #word_list = individual_Review.split("<br /><br") #'<br ><br >'
    word_list = individual_Review.strip().split(" ")
    #print(word_list)
    D = {}
    for word in word_list:
        word.lower()
        #print(word)
        #String processing bullshit
        raw_word = ""
        for letter in word:

            if (letter == 'a' or letter == 'b' or letter == 'c' or letter == 'd' or letter == 'e' or letter == 'f' or letter == 'g' or letter == 'h' or
            letter == 'i' or letter == 'j' or letter == 'k' or letter == 'l' or letter == 'm' or letter == 'n' or letter == 'o' or letter == 'p' or
            letter == 'q' or letter == 'r' or letter == 's' or letter == 't' or letter == 'u' or letter == 'v' or letter == 'w' or letter == 'x' or
            letter == 'y' or letter == 'z'):

                raw_word = raw_word + letter
            
#         '''
#         for word in word_list:
#             if word.isalpha():
#                 raw_word += word.lower()
#         #print(raw_word)             
        #if set(raw_word):      
        if raw_word in D:
            D[raw_word] += 1
        else:
            D[raw_word] = 1
# '''
            
    individual_Review_sum = sum(D.values())
    #print("individual review sum ..." + str(individual_Review_sum))
    #print(D)
    return D, individual_Review_sum


def Total_Reviews__word_Sum(total_reviews):
    #Iterate through all reviews one review at a time
    #Iterate through each word in review and add key values
    '''
    reviews_sum = 0
    for review in total_reviews:
        for word in review:
           reviews_sum = review[word] + reviews_sum
    return reviews_sum
    '''
    Reviews_list = []
    Reviews_dict = {}

    word_sum = 0
    for review in total_reviews:
        for word in review:
            if (word in Reviews_dict):
                Reviews_dict[word] = Reviews_dict[word] + 1
            else:
                Reviews_dict[word] = 1
        Reviews_list.append(Reviews_dict)
        word_sum = sum(Reviews_dict.values()) + word_sum
    return word_sum


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

def get_num_of_word(Reviews):
    num_of_word = {}

    for review in Reviews:
        for word in review:
            if (word in num_of_word):
                num_of_word[word] = num_of_word[word] + 1
            else:
                num_of_word[word] = 1
        
    #print(num_of_word)
    return num_of_word



def Gaussian_BOW(Pos_Reviews, Neg_Reviews, Pos_count, Neg_count):
    '''
    number of times that word appears in a negative + some alpha value
    _________________________________________________________________
    total words in that negative + some alpha * (total words in negative + positive)
    '''
    #setting alpha value equal to 1
    alpha = 1
    BOW_dict_pos = {}
    BOW_dict_neg = {}
    neg_sum = sum(Neg_count)
    pos_sum = sum(Pos_count)

    pos_word_sum = get_num_of_word(Pos_Reviews)
    neg_word_sum = get_num_of_word(Neg_Reviews)

    #do pos prob first
    for Review in Pos_Reviews:
        for word in Review:
            desired_pos_word = pos_word_sum[word]
            pos_numerator = desired_pos_word + alpha
            pos_denominator = (neg_sum + (alpha * (neg_sum + pos_sum)))
            pos_word_prob = math.log((pos_numerator / pos_denominator))           
            BOW_dict_pos[word] = pos_word_prob
            #print(pos_word_prob)
    
    #do neg prob now
    for Review in Neg_Reviews:
        for word in Review:
            desired_neg_word = neg_word_sum[word]
            neg_numerator = desired_neg_word + alpha
            neg_denominator = (neg_sum + (alpha * (neg_sum + pos_sum)))
            neg_word_prob = math.log((neg_numerator / neg_denominator))              
            BOW_dict_neg[word] = neg_word_prob
            #print(neg_word_prob)

    return BOW_dict_pos, BOW_dict_neg

def compare_public_vs_training(Pos_Reviews, Neg_Reviews, Pos_count, Neg_count):
    public_pos, pos_count = open_File_train_and_extract_review('test_pos_public.txt')
    public_neg, neg_count = open_File_train_and_extract_review('test_neg_public.txt')

    Pos_Train_BOW_Prob, Neg_Train_BOW_Prob = Gaussian_BOW(Pos_Reviews, Neg_Reviews, Pos_count, Neg_count)

    Public_POS_Classification_P = 0
    Public_NOS_Classification_N = 0
    pos_count = 0
    neg_count = 0
    errors = 0

    for Review in public_pos:
        for word in Review:
            try:
                if(Pos_Train_BOW_Prob[word]):
                    Public_POS_Classification_P = Pos_Train_BOW_Prob[word] + Public_POS_Classification_P
                if( Neg_Train_BOW_Prob[word]):
                    Public_NOS_Classification_N = Neg_Train_BOW_Prob[word] + Public_NOS_Classification_N
            except KeyError:
                errors = errors + 1
                #if(Neg_Train_BOW_Prob[word]):
                #        Public_NOS_Classification_N = Neg_Train_BOW_Prob[word] + Public_NOS_Classification_N
                #    if(Pos_Train_BOW_Prob[word]):
                #       Public_POS_Classification_P = Pos_Train_BOW_Prob[word] + Public_POS_Classification_P
            if(Public_NOS_Classification_N > Public_POS_Classification_P):
                Pos_count = pos_count + 1
            else:
                neg_count = neg_count + 1  

    Pos_Accuracy = ((pos_count) / len(public_pos)) * 100
    Neg_Accuracy = ((neg_count) / len(public_neg)) * 100
    print("POS Accuracy is ... " + str(Pos_Accuracy))
    print("NEG Accuracy is ..." + str(Neg_Accuracy))





def main():
    #returns vector list -> Reviews
    #returns dictionary -> count_word
    Positive_Reviews, Positive_count_word = open_File_train_and_extract_review('test_pos_public.txt')
    Negative_Reviews, Negative_count_word = open_File_train_and_extract_review('test_neg_public.txt')
    #Negative_Reviews, Negative_count_word = open_negFile_train_and_extract_review()

    #print(Positive_Reviews)
    #print(Positive_count_word)

    # print(Positive_count_word)
    # a = sum(Positive_count_word)
    # print(a)

    # print(sum(get_num_of_word(Positive_Reviews)))
    # print(sum(Positive_count_word))
    # print(sum(get_num_of_word(Negative_Reviews)))
    # print(sum(Negative_count_word))


    #Gaussian_BOW(Positive_Reviews, Negative_Reviews, Positive_count_word, Negative_count_word)
    compare_public_vs_training(Positive_Reviews, Negative_Reviews, Positive_count_word, Negative_count_word)

    #TF = computeTF(Reviews, count_word)
    #IDF = computeIDF(Reviews, count_word)
    #TF_IDF = computeTF_IDF(TF, IDF)


    #Bayes Theorem 
    '''Assume Each data point is independent'''
    '''
                    P(B|A) * P(A)
     P(A|B) =       ______________
                        P(B)
    '''
    #Each word in the review will be a feature



    #print(len(TF_IDF))

if __name__ == '__main__':
    main()



