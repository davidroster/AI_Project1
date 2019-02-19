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
        temp = 0

        for review in reviews:
            #Pushes in a list of strings
            D, individual_review_sum,sum_of_keys  = vectorize(review)
            temp = temp + sum_of_keys
            vector_list.append(D)
            review_sum[count_sum] = individual_review_sum
            count_sum = count_sum + 1

    #Returns complete vector to main function
    return  vector_list, review_sum,temp

def vectorize(individual_Review):
    #Takes in a list of strings
    #word_list = individual_Review.split("<br /><br") #'<br ><br >'
    word_list = individual_Review.strip().split(" ")
    #print(word_list)
    D = {}
    word_count = 0
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
        word_count = word_count + 1
# '''
            
    individual_Review_sum = sum(D.values())
    #sum_of_keys = sum(D.keys())
    #print("individual review sum ..." + str(individual_Review_sum))
    #print(D)
    return D, individual_Review_sum, word_count


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



#def Gaussian_BOW(Pos_Reviews, Neg_Reviews, Pos_count, Neg_count):
def Gaussian_BOW(word_sum, total_keys, word, classification_keys):
    '''
    number of times that word appears in a negative + some alpha value
    _________________________________________________________________
    total words in that negative + some alpha * (total words in negative + positive vocab (*unique words*))
    '''
    
    alpha = 1
    desired_word = word_sum[word]
    numerator = desired_word + alpha
    #denominator = (classification_keys + (alpha * (neg_number_of_keys + pos_number_of_keys)))
    denominator = (classification_keys + (alpha * (total_keys)))
    word_prob = math.log((numerator / denominator))           


    #trying something
    '''
    #setting alpha value equal to 1
    alpha = 1
    BOW_dict_pos = {}
    BOW_dict_neg = {}
    #neg_sum = sum(Neg_count)
    #pos_sum = sum(Pos_count)
    neg_sum = Neg_count #neg count is training number of neg keys
    pos_sum = Pos_count #pos count is training number of pos keys

    #do pos prob first
    for Review in Pos_Reviews:
        for word in Review:
            if(word not in BOW_dict_pos):
                desired_pos_word = pos_word_sum[word]
                pos_numerator = desired_pos_word + alpha
                pos_denominator = (pos_sum + (alpha * (neg_sum + pos_sum)))
                pos_word_prob = math.log((pos_numerator / pos_denominator))           
                BOW_dict_pos[word] = pos_word_prob
                #print(pos_word_prob)
    
    #do neg prob now
    for Review in Neg_Reviews:
        for word in Review:
            if(word not in BOW_dict_neg):
                desired_neg_word = neg_word_sum[word]
                neg_numerator = desired_neg_word + alpha
                neg_denominator = (neg_sum + (alpha * (neg_sum + pos_sum)))
                neg_word_prob = math.log((neg_numerator / neg_denominator))              
                BOW_dict_neg[word] = neg_word_prob
                #print(neg_word_prob)

    return BOW_dict_pos, BOW_dict_neg
    '''
    word_prob = 1
    return word_prob

def compare_public_vs_training(Pos_Reviews, Neg_Reviews, Pos_count, Neg_count, train_pos_number_keys, train_neg_number_keys):
    public_pos, pos_count, sum_of_public_pos_keys = open_File_train_and_extract_review('test_pos_public.txt')
    public_neg, neg_count, sum_of_public_neg_keys = open_File_train_and_extract_review('test_neg_public.txt')

    #Pos_Train_BOW_Prob, Neg_Train_BOW_Prob = Gaussian_BOW(Pos_Reviews, Neg_Reviews, Pos_count, Neg_count)
    '''Pos_Train_BOW_Prob, Neg_Train_BOW_Prob = Gaussian_BOW(Pos_Reviews, Neg_Reviews, train_pos_number_keys, train_neg_number_keys)'''

    Public_POS_Classification_P = 0
    Public_NOS_Classification_N = 0
    posPublic_pos_count = 0
    posPublic_neg_count = 0
    negPublic_neg_count = 0
    negPublic_pos_count = 0

    total_keys = train_neg_number_keys + train_pos_number_keys
    pos_word_count = get_num_of_word(Pos_Reviews)
    neg_word_count = get_num_of_word(Neg_Reviews)
    temp_pos = train_pos_number_keys
    temp_neg = train_neg_number_keys

    for Review in public_pos:
        Public_POS_Classification_N = 0
        Public_POS_Classification_P = 0  
        for word in Review:
            try:
                Pos_Train_BOW_Prob = Gaussian_BOW(pos_word_count, total_keys, word, train_pos_number_keys)
                Public_POS_Classification_P = Pos_Train_BOW_Prob + Public_POS_Classification_P

            except KeyError:
                Public_POS_Classification_P = Public_POS_Classification_P + 0
            
            try:
                Neg_Train_BOW_Prob = Gaussian_BOW(neg_word_count, total_keys, word, train_neg_number_keys)
                Public_POS_Classification_N = Neg_Train_BOW_Prob + Public_POS_Classification_N
                
            except KeyError:
                Public_POS_Classification_N = Public_POS_Classification_N + 0
                
        if(Public_POS_Classification_P > Public_POS_Classification_N):
            posPublic_pos_count = posPublic_pos_count + 1
            print("pos count is ..." + str(posPublic_pos_count))
        else:
        #if(Public_NOS_Classification_N > Public_POS_Classification_P):
            posPublic_neg_count = posPublic_neg_count + 1
            #print("neg count is ..." + str(posPublic_neg_count))

    Pos_Public_Pos_Accuracy = ((posPublic_pos_count) / (posPublic_pos_count + posPublic_neg_count)) * 100
    Pos_Public_Neg_Accuracy = ((posPublic_neg_count) / (posPublic_pos_count + posPublic_neg_count)) * 100
    print("Positive class -> POS Accuracy is ... " + str(Pos_Public_Pos_Accuracy))
    print("Positive class -> NEG Accuracy is ..." + str(Pos_Public_Neg_Accuracy))

    for Review in public_neg:
        Public_NEG_Classification_N = 0
        Public_NEG_Classification_P = 0  
        for word in Review:
            try:
                Pos_Train_BOW_Prob = Gaussian_BOW(pos_word_count, total_keys, word, train_pos_number_keys)
                Public_NEG_Classification_P = Pos_Train_BOW_Prob + Public_NEG_Classification_P

            except KeyError:
                Public_NEG_Classification_P = Public_NEG_Classification_P + 0
            
            try:
                Neg_Train_BOW_Prob = Gaussian_BOW(neg_word_count, total_keys, word, train_neg_number_keys)
                Public_NEG_Classification_N = Neg_Train_BOW_Prob + Public_NEG_Classification_N
                
            except KeyError:
                Public_NEG_Classification_N = Public_NEG_Classification_N + 0
                
        if(Public_POS_Classification_P > Public_POS_Classification_N):
            negPublic_pos_count = negPublic_pos_count + 1
            #print("pos count is ..." + str(negPublic_pos_count))
        else:
        #if(Public_NOS_Classification_N > Public_POS_Classification_P):
            negPublic_neg_count = negPublic_neg_count + 1
            #print("neg count is ..." + str(negPublic_neg_count))

    Neg_Public_Pos_Accuracy = ((negPublic_pos_count) / (negPublic_pos_count + negPublic_neg_count)) * 100
    Neg_Public_Neg_Accuracy = ((negPublic_neg_count) / (negPublic_pos_count + negPublic_neg_count)) * 100
    print("NEG PUBLIC -> POS Accuracy is ... " + str(Neg_Public_Pos_Accuracy))
    print("NEG PUBLIC -> NEG Accuracy is ..." + str(Neg_Public_Neg_Accuracy))

def main():
    #returns vector list -> Reviews
    #returns dictionary -> count_word
    Positive_Reviews, Positive_count_word, pos_key_count = open_File_train_and_extract_review('test_pos_public.txt')
    Negative_Reviews, Negative_count_word, neg_key_count = open_File_train_and_extract_review('test_neg_public.txt')

    compare_public_vs_training(Positive_Reviews, Negative_Reviews, Positive_count_word, Negative_count_word, pos_key_count, neg_key_count)

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

if __name__ == '__main__':
    main()



