#David Roster
#Programming Assignment #1

import sys  
import os
import string
import math
import statistics
#import numpy as np

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
        # temp = ""
        # for line in fp:
        temp = fp.read()
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
    word_list = individual_Review.strip().split(" ")
    D = {}
    Stop_Word_D = {"a", "about", "above","after","again","against","all","a","man","and", "any", "area", "sat","be","because",
                    "been","before","being","below","between","both","but","by","could","did","do","does","doing","down","during",
                    "each","few","for","from","further","had","has","have","having","he","hed","hes","her","here","heres","self","him",
                    "himself","his","how","hows","sit","sits","more","most","my","myself","no","other","ought","our","oursourselves",
                    "out","over","own","same","such","than","that","theirs","themthemselves","then","there","theres","these","they",
                    "theyd","theyll","theyre","theyve","this","what","whats","when","where","wheres","which","while","who","why"}
    word_count = 0
    for word in word_list:
        word.lower()
        #String processing bullshit
        raw_word = ""
        for letter in word:

            if (letter == 'a' or letter == 'b' or letter == 'c' or letter == 'd' or letter == 'e' or letter == 'f' or letter == 'g' or letter == 'h' or
            letter == 'i' or letter == 'j' or letter == 'k' or letter == 'l' or letter == 'm' or letter == 'n' or letter == 'o' or letter == 'p' or
            letter == 'q' or letter == 'r' or letter == 's' or letter == 't' or letter == 'u' or letter == 'v' or letter == 'w' or letter == 'x' or
            letter == 'y' or letter == 'z'):

                raw_word = raw_word + letter
        
        if(raw_word):
            if(raw_word not in Stop_Word_D):     
                if raw_word in D:
                    D[raw_word] += 1
                else:
                    D[raw_word] = 1
                    word_count = word_count + 1
            
    individual_Review_sum = sum(D.values())
    return D, individual_Review_sum, word_count

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


#def Multivariative_BOW(Pos_Reviews, Neg_Reviews, Pos_count, Neg_count):
def Multivariative_BOW(word_sum, total_keys, word, classification_keys):
    '''
    number of times that word appears in a negative + some alpha value
    _________________________________________________________________
    total words in that negative + some alpha * (total words in negative + positive vocab (*unique words*))
    '''
    
    alpha = 1
    desired_word = word_sum[word]
    numerator = desired_word + alpha
    denominator = (classification_keys + (alpha * (total_keys)))
    word_prob = math.log((numerator / denominator))           

    return word_prob

def compare_public_vs_training(Pos_Reviews, Neg_Reviews, Pos_count, Neg_count, train_pos_number_keys, train_neg_number_keys):
    public_pos, pos_count, sum_of_public_pos_keys = open_File_train_and_extract_review(sys.argv[3])
    public_neg, neg_count, sum_of_public_neg_keys = open_File_train_and_extract_review(sys.argv[4])

    posPublic_pos_count = 0
    posPublic_neg_count = 0
    negPublic_neg_count = 0
    negPublic_pos_count = 0

    number_of_class_keys_pos = len(Pos_count)
    number_of_class_keys_neg = len(Neg_count)

    total_keys = train_neg_number_keys + train_pos_number_keys
    pos_word_count = get_num_of_word(Pos_Reviews)
    neg_word_count = get_num_of_word(Neg_Reviews)

    for Review in public_pos:
        Public_POS_Classification_N = 0
        Public_POS_Classification_P = 0  
        for word in Review:
            try:
                Pos_Train_BOW_Prob = Multivariative_BOW(pos_word_count, total_keys, word, number_of_class_keys_pos)
                Public_POS_Classification_P = Pos_Train_BOW_Prob + Public_POS_Classification_P

            except KeyError:
                Public_POS_Classification_P = Public_POS_Classification_P + 0
            
            try:
                Neg_Train_BOW_Prob = Multivariative_BOW(neg_word_count, total_keys, word, number_of_class_keys_neg)
                Public_POS_Classification_N = Neg_Train_BOW_Prob + Public_POS_Classification_N
                
            except KeyError:
                Public_POS_Classification_N = Public_POS_Classification_N + 0
                
        if(Public_POS_Classification_P > Public_POS_Classification_N):
            posPublic_pos_count = posPublic_pos_count + 1
        else:
            posPublic_neg_count = posPublic_neg_count + 1

    Pos_Public_Pos_Accuracy = ((posPublic_pos_count) / (posPublic_pos_count + posPublic_neg_count)) * 100
    Pos_Public_Neg_Accuracy = ((posPublic_neg_count) / (posPublic_pos_count + posPublic_neg_count)) * 100
    print("Positive class -> POS Accuracy is ... " + str(Pos_Public_Pos_Accuracy))
    print("Positive class -> NEG Accuracy is ..." + str(Pos_Public_Neg_Accuracy))


    for Review in public_neg:
        Public_NEG_Classification_N = 0
        Public_NEG_Classification_P = 0  
        for word in Review:
            try:
                Pos_Train_BOW_Prob = Multivariative_BOW(pos_word_count, total_keys, word, number_of_class_keys_pos)
                Public_NEG_Classification_P = Pos_Train_BOW_Prob + Public_NEG_Classification_P

            except KeyError:
                Public_NEG_Classification_P = Public_NEG_Classification_P + 0
            
            try:
                Neg_Train_BOW_Prob = Multivariative_BOW(neg_word_count, total_keys, word, number_of_class_keys_neg)
                Public_NEG_Classification_N = Neg_Train_BOW_Prob + Public_NEG_Classification_N
                
            except KeyError:
                Public_NEG_Classification_N = Public_NEG_Classification_N + 0
                
        if(Public_NEG_Classification_P > Public_NEG_Classification_N):
            negPublic_pos_count = negPublic_pos_count + 1
        else:
            negPublic_neg_count = negPublic_neg_count + 1

    Neg_Public_Pos_Accuracy = ((negPublic_pos_count) / (negPublic_pos_count + negPublic_neg_count)) * 100
    Neg_Public_Neg_Accuracy = ((negPublic_neg_count) / (negPublic_pos_count + negPublic_neg_count)) * 100
    print("Negative class -> POS Accuracy is ... " + str(Neg_Public_Pos_Accuracy))
    print("Negative class -> NEG Accuracy is ..." + str(Neg_Public_Neg_Accuracy))

















def Format_Gaussian_BOW(Reviews):
    print("Entering Format_Gaussian_BOW Function")
    print("Lets get it")

    #Stores index values of words in a List of Dictionaries
    Gaussian_BOW_Unique_WordList = []
    #Stores total amount of each word in the .txt file in a Dictionary
    Gaussian_BOW_TotalWords_Dict = {}

    Gaussian_BOW_WordCount_INReviews_WordList =[]
    Gaussian_BOW_WordCount_IN_Individual_Review_Dict = {}

    Gaussian_NumberofReviews_a_WordIsIN_Dict = {}

    words_in_each_review_dict = {}
    count = 0

    for review_placeholder, review in enumerate(Reviews):
        Word_with_index_value_Dict = {}
        index = 0
        for word in review:
            count = count + 1

            #displays 
            if(word in Gaussian_BOW_WordCount_IN_Individual_Review_Dict):
                Gaussian_BOW_WordCount_IN_Individual_Review_Dict[word] = Gaussian_BOW_WordCount_IN_Individual_Review_Dict[word] + 1
            else:
                Gaussian_BOW_WordCount_IN_Individual_Review_Dict[word] = 1

            if(word in Gaussian_BOW_TotalWords_Dict):
                Gaussian_BOW_TotalWords_Dict[word] = Gaussian_BOW_TotalWords_Dict[word] + 1
            else:
                Gaussian_BOW_TotalWords_Dict[word] = 1

            if(word not in Word_with_index_value_Dict):
                Word_with_index_value_Dict[word] = index
                index = index + 1

        words_in_each_review_dict[review_placeholder] = count
            
            
        #print("Word_with_index_value_Dict", Word_with_index_value_Dict)
        Gaussian_BOW_Unique_WordList.append(Word_with_index_value_Dict)
        Gaussian_BOW_WordCount_INReviews_WordList.append(Gaussian_BOW_WordCount_IN_Individual_Review_Dict)
    #print("Gaussian_BOW_TotalWords_Dict", Gaussian_BOW_TotalWords_Dict)
    return Gaussian_BOW_Unique_WordList, Gaussian_BOW_WordCount_INReviews_WordList, Gaussian_BOW_TotalWords_Dict, words_in_each_review_dict


def Gaussian_BOW(Pos_Train_Review, Neg_Train_Review):
    #Pos_Public_Review, pos_count, sum_of_public_pos_keys = open_File_train_and_extract_review(sys.argv[3])
    #Neg_Public_Review, neg_count, sum_of_public_neg_keys = open_File_train_and_extract_review(sys.argv[4])
    
    #return order for Format_Gaussian_BOW
        #Gaussian_BOW_Unique_WordList -> #Stores index values of words in a List of Dictionaries
        #Gaussian_BOW_WordCount_INReviews_WordList -> number of times each word appears in a review
        #Gaussian_BOW_TotalWords_Dict ->   #Stores total amount of each word in the .txt file in a Dictionary

    Format_Gaussian_Pos_Train_Review, Pos_WordCount_INReviews_WordList, Pos_TotalWords_Dict, pos_words_in_each_review_dict = Format_Gaussian_BOW(Pos_Train_Review)
    Format_Gaussian_Neg_Train_Review, Neg_WordCount_INReviews_WordList, Neg_TotalWords_Dict, neg_words_in_each_review_dict = Format_Gaussian_BOW(Neg_Train_Review)

    Pos_mean_dict = {}
    Pos_stdev_dict = {}
    Neg_mean_dict = {}
    Neg_stdev_dict = {}
    
    review_count = 0
    index_counter =0

    print(len(Pos_Train_Review))
    print(len(Format_Gaussian_Pos_Train_Review))

    #matrix = [len(Pos_Train_Review) - 1][len(Format_Gaussian_Pos_Train_Review) - 1]
    matrix = [[0 for i in range(len(Pos_Train_Review))] for j in range(len(Format_Gaussian_Pos_Train_Review))]
    #if i can get numpy do this bright below and comment out matrix
    #matrix = numpy.zeros(len(Pos_Train_Review), len(Format_Gaussian_Pos_Train_Review))

    for index,review in enumerate(Pos_Train_Review):
        indexDict = {}
        for word in review:
            if(word not in indexDict):
                indexDict[word] = index_counter
                index_counter = index_counter + 1
                word_value = Pos_WordCount_INReviews_WordList[index][word]
                matrix[review_count][indexDict[word]] = word_value
        review_count = review_count + 1

    print(matrix)

    

    '''
    print("Entering main For Loop in Gaussian BOW")
    for review in Format_Gaussian_Pos_Train_Review:
        temp_List = []
        print("Entering Format_Gaussian_Pos_Train_Review For Loop in Gaussian BOW")
        for word in review:
            try:
                if(review[word]):
                    temp = review[word]
                    temp_List.append(temp)
            except KeyError:
                temp_List.append(0)


        mean = statistics.mean(temp_List)
        stdev = statistics.stdev(temp_List)
        print("The current word is ...", word)
        print("mean of this word is ... ", mean)
        print("Stdev of this word is ...",stdev)
        mean_dict[word] = mean
        stdev_dict[word] = stdev
    
    print(mean_dict)
    print(stdev_dict)
    '''
    return Pos_mean_dict, Pos_stdev_dict, Neg_mean_dict, Neg_stdev_dict, pos_words_in_each_review_dict, neg_words_in_each_review_dict



def GNB(word, word_value, mean_dict, stdev_dict):
    #x = number of times you saw word in review / total words in that review 

    mean = mean_dict[word]
    stdev = stdev_dict[word]
    x = word_value

    if (stdev != 0):
        Gaussian_Naive_Bayes = math.log(1.0/float(math.sqrt(2*3.1415926*(stdev**2))) - (float((x - mean)**2)/float(2*stdev)))
    else:
        Gaussian_Naive_Bayes = 0

    return Gaussian_Naive_Bayes



def Gaussian_Formula_Classification(Pos_Train_Reviews, Neg_Train_Reviews, Pos_Count_Word, Neg_Count_Word):
    public_pos, pos_count, sum_of_public_pos_keys = open_File_train_and_extract_review(sys.argv[3])
    public_neg, neg_count, sum_of_public_neg_keys = open_File_train_and_extract_review(sys.argv[4])

    Pos_Mean_dict, Pos_stdev_dict, Neg_Mean_dict, Neg_stdev_dict, Pos_WordCount_INReviews_dict, Neg_WordCount_INReviews_dict = Gaussian_BOW(Pos_Train_Reviews, Neg_Train_Reviews)

    review_number_inc = 1
    total_Gaussian_Probability = 0

    for review in public_pos:
        for word in review:
            try:
                review_value = Pos_Count_Word[review_number_inc]
                number_of_words_in_currentreview = Pos_WordCount_INReviews_dict[review_number_inc]
                word_value = review_value / number_of_words_in_currentreview
                Gaussian_Prob = GNB(word, word_value, Pos_Mean_dict, Pos_stdev_dict)
            except KeyError:
                Gaussian_Prob = Gaussian_Prob + 0

        review_number_inc = review_number_inc + 1

    # for Review in public_pos:
    #     Public_POS_Classification_N = 0
    #     Public_POS_Classification_P = 0  
    #     for word in Review:
    #         try:
    #             Pos_Train_BOW_Prob = Gaussian_BOW(Pos_Train_Reviews, Neg_Train_Reviews)
    #             Public_POS_Classification_P = Pos_Train_BOW_Prob + Public_POS_Classification_P

    #         except KeyError:
    #             Public_POS_Classification_P = Public_POS_Classification_P + 0
            
    #         try:
    #             Neg_Train_BOW_Prob = Gaussian_BOW(neg_word_count, total_keys, word, number_of_class_keys_neg)
    #             Public_POS_Classification_N = Neg_Train_BOW_Prob + Public_POS_Classification_N
                
    #         except KeyError:
    #             Public_POS_Classification_N = Public_POS_Classification_N + 0
                
    #     if(Public_POS_Classification_P > Public_POS_Classification_N):
    #         posPublic_pos_count = posPublic_pos_count + 1
    #     else:
    #         posPublic_neg_count = posPublic_neg_count + 1

    # Pos_Public_Pos_Accuracy = ((posPublic_pos_count) / (posPublic_pos_count + posPublic_neg_count)) * 100
    # Pos_Public_Neg_Accuracy = ((posPublic_neg_count) / (posPublic_pos_count + posPublic_neg_count)) * 100
    # print("Positive class -> POS Accuracy is ... " + str(Pos_Public_Pos_Accuracy))
    # print("Positive class -> NEG Accuracy is ..." + str(Pos_Public_Neg_Accuracy))



def main():

    if len(sys.argv) == 5:
        #returns vector list -> Reviews
        #returns dictionary -> count_word
        Positive_Reviews, Positive_count_word, pos_key_count = open_File_train_and_extract_review(sys.argv[1])
        Negative_Reviews, Negative_count_word, neg_key_count = open_File_train_and_extract_review(sys.argv[2])

    #compare_public_vs_training(Positive_Reviews, Negative_Reviews, Positive_count_word, Negative_count_word, pos_key_count, neg_key_count)

    #Gaussian_BOW(Positive_Reviews, Negative_Reviews)

    Gaussian_Formula_Classification(Positive_Reviews, Negative_Reviews, Positive_count_word, Negative_count_word)

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



