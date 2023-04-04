# import gc
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import random
from fuzzywuzzy import fuzz
from nltk.util import ngrams
import warnings
warnings.filterwarnings('ignore')

def data_visualization(df):
    # pass the data as a dataframe
    # data visualization
    print(df.head())

    # cheack if any null values
    print(df.isnull().sum())

    #check for the duplicate
    print(df.duplicated().sum())

    # check for how many total no of duplicates and total number of not duplicate
    print(df['is_duplicate'].value_counts())
    print((df['is_duplicate'].value_counts()/df['is_duplicate'].count())*100)
    df['is_duplicate'].value_counts().plot(kind='bar')
    plt.savefig('total_dup_nondup.png')

    #check how many duplicate questions are there

    qid = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
    print('Number of unique questions',np.unique(qid).shape[0])
    x = qid.value_counts()>1
    print('Number of questions getting repeated',x[x].shape[0])

    # Repeated questions histogram

    plt.hist(qid.value_counts().values,bins=160)
    plt.yscale('log')
    plt.savefig('each_que_no_rep.png')

def data_engineering(data):
    # adding the length (each indivuadial letters) of each questions to the data frame
    data['q1_len'] = data['question1'].str.len() 
    data['q2_len'] = data['question2'].str.len()

    # adding the total number of words of each questions in the data frame
    data['q1_num_words'] = data['question1'].apply(lambda row: len(str(row).split(" ")))
    data['q2_num_words'] = data['question2'].apply(lambda row: len(str(row).split(" ")))

    # common_words
    data['overlap_count'] = (data.apply(lambda r: set(r['question1'].split(" ")) &
                                             set(r['question2'].split(" ")),                               
                                            axis=1)).str.len()

    # count the total number of words from both question 1 and 2
    data['total_num_words'] = data['q1_num_words']+data['q2_num_words']

    data['word_share'] = round(data['overlap_count']/data['total_num_words'],2)

    # finding common_word_count min and max, common_stop_count min and max, common_token_count min and max, last_word_eq, first_word_eq
    def fetch_token_features(row):
        
        q1 = row['question1']
        q2 = row['question2']

        SAFE_DIV = 0.0001 

        STOP_WORDS = stopwords.words("english")
        
        token_features = [0.0]*8
        
        # Converting the Sentence into Tokens: 
        q1_tokens = q1.split()
        q2_tokens = q2.split()
        
        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return token_features

        # Get the non-stopwords in Questions
        q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
        
        #Get the stopwords in Questions
        q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
        q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
        
        # Get the common non-stopwords from Question pair
        common_word_count = len(q1_words.intersection(q2_words))
        
        # Get the common stopwords from Question pair
        common_stop_count = len(q1_stops.intersection(q2_stops))
        
        # Get the common Tokens from Question pair
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
        
        
        token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
        token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
        token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        
        # Last word of both question is same or not
        token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
        
        # First word of both question is same or not
        token_features[7] = int(q1_tokens[0] == q2_tokens[0])
        
        return token_features

    token_features = data.apply(fetch_token_features, axis=1)

    data["cwc_min"]       = list(map(lambda x: x[0], token_features))
    data["cwc_max"]       = list(map(lambda x: x[1], token_features))
    data["csc_min"]       = list(map(lambda x: x[2], token_features))
    data["csc_max"]       = list(map(lambda x: x[3], token_features))
    data["ctc_min"]       = list(map(lambda x: x[4], token_features))
    data["ctc_max"]       = list(map(lambda x: x[5], token_features))
    data["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    data["first_word_eq"] = list(map(lambda x: x[7], token_features))

    # finding the absoulute length diff, mean len, and longest subtr ratio
    def fetch_length_features(row):
        q1 = row['question1']
        q2 = row['question2']
        
        length_features = [0.0]*3
        
        # Converting the Sentence into Tokens: 
        q1_tokens = q1.split()
        q2_tokens = q2.split()
        
        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return length_features
        
        # Absolute length features
        length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
        
        #Average Token Length of both Questions
        length_features[1] = (len(q1_tokens) + len(q2_tokens))/2
        
        return length_features

    length_features = data.apply(fetch_length_features, axis=1)

    data['abs_len_diff'] = list(map(lambda x: x[0], length_features))
    data['mean_len'] = list(map(lambda x: x[1], length_features))
    data['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))

    # fuzzy features

    def fetch_fuzzy_features(row):
    
        q1 = row['question1']
        q2 = row['question2']
        
        fuzzy_features = [0.0]*4
        
        # fuzz_ratio
        fuzzy_features[0] = fuzz.QRatio(q1, q2)

        # fuzz_partial_ratio
        fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

        # token_sort_ratio
        fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

        # token_set_ratio
        fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

        return fuzzy_features

    fuzzy_features = data.apply(fetch_fuzzy_features, axis=1)

    # Creating new feature columns for fuzzy features
    data['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
    data['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
    data['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
    data['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))

    return data

def data_analysis(dataset):

    # Analysis the features of q1_len
    sns.displot(dataset['q1_len'])
    print('minimum characters',dataset['q1_len'].min())
    print('maximum characters',dataset['q1_len'].max())
    print('average num of characters',int(dataset['q1_len'].mean()))
    plt.savefig('q1_len_data_analysis.png')

    # Analysis the features of q2_len
    sns.displot(dataset['q2_len'])
    print('minimum characters',dataset['q2_len'].min())
    print('maximum characters',dataset['q2_len'].max())
    print('average num of characters',int(dataset['q2_len'].mean()))
    plt.savefig('q2_len_data_analysis.png')

    # Analysis the features of q1_num_words
    sns.displot(dataset['q1_num_words'])
    print('minimum words',dataset['q1_num_words'].min())
    print('maximum words',dataset['q1_num_words'].max())
    print('average num of words',int(dataset['q1_num_words'].mean()))
    plt.savefig('q1_num_words_data_analysis.png')

    # Analysis the features of q2_num_words
    sns.displot(dataset['q2_num_words'])
    print('minimum words',dataset['q2_num_words'].min())
    print('maximum words',dataset['q2_num_words'].max())
    print('average num of words',int(dataset['q2_num_words'].mean()))
    plt.savefig('q2_num_words_data_analysis.png')

    # common words
    sns.distplot(dataset[dataset['is_duplicate'] == 0]['overlap_count'],label='non duplicate')
    sns.distplot(dataset[dataset['is_duplicate'] == 1]['overlap_count'],label='duplicate')
    plt.savefig('common_words_data_analysis.png')

    # total words
    sns.distplot(dataset[dataset['is_duplicate'] == 0]['total_num_words'],label='non duplicate')
    sns.distplot(dataset[dataset['is_duplicate'] == 1]['total_num_words'],label='duplicate')
    plt.savefig('total_words_data_analysis.png')

    # word share
    sns.distplot(dataset[dataset['is_duplicate'] == 0]['word_share'],label='non duplicate')
    sns.distplot(dataset[dataset['is_duplicate'] == 1]['word_share'],label='duplicate')
    plt.savefig('word_share_data_analysis.png')
    # print(dataset.head())

    # comparision image between common token, word, stop word count min
    sns.pairplot(dataset[['ctc_min', 'cwc_min', 'csc_min', 'is_duplicate']],hue='is_duplicate')
    plt.savefig('comparision_between_ctc_cwc_csc_min.png')

    # comparision image between common token, word, stop word count max
    sns.pairplot(dataset[['ctc_max', 'cwc_max', 'csc_max', 'is_duplicate']],hue='is_duplicate')
    plt.savefig('comparision_between_ctc_cwc_csc_max.png')

    # comparision image between last word common equal, first word equal
    sns.pairplot(dataset[['last_word_eq', 'first_word_eq', 'is_duplicate']],hue='is_duplicate')
    plt.savefig('comparision_lastw_firstw.png')

    # comparision image between mean length, abs len, longest substring ratio
    sns.pairplot(dataset[['mean_len', 'abs_len_diff','longest_substr_ratio', 'is_duplicate']],hue='is_duplicate')
    plt.savefig('comparision_meanlen_abslen_lgstsbstrat.png')

    # comparision image between fuzzy ratio, fuzzy partial ratio, token sort ratio and token set ratio
    sns.pairplot(dataset[['fuzz_ratio', 'fuzz_partial_ratio','token_sort_ratio','token_set_ratio', 'is_duplicate']],hue='is_duplicate')
    plt.savefig('comparision_between_fuzzy.png')

def ngram_features(row):
    
    # Get the two questions from the row
    q1 = row['question1']
    q2 = row['question2']

    # Define a small constant for safe division
    SAFE_DIV = 0.0001 

    # Define a set of stop words
    STOP_WORDS = stopwords.words("english")
    
    # Initialize a list for the n-gram features
    ngram_feature = [0.0]*2
    
    # Convert the questions into tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    # Extract the bigrams from the questions
    bigram_que1 = set(ngrams(q1_tokens, 2)) 
    bigram_que2 = set(ngrams(q2_tokens, 2)) 

    # Remove stop words from the questions
    q1_words = set(word for word in q1_tokens if word not in STOP_WORDS)
    q2_words = set(word for word in q2_tokens if word not in STOP_WORDS)
    
    # Compute the number of common bigrams between the two questions
    count_of_bigram1 = len(bigram_que1.intersection(q2_words))
    count_of_bigram2 = len(bigram_que2.intersection(q1_words))
    
    # Compute the Jaccard similarity between the two sets of non-stop words
    jaccard_sim = (float(len(q1_words.intersection(q2_words))) / 
                   (len(q1_words.union(q2_words)) + SAFE_DIV))
    
    # Add the n-gram features to the list
    ngram_feature[0] = count_of_bigram1
    ngram_feature[1] = count_of_bigram2
    
    # Return the list of n-gram features
    return ngram_feature + [jaccard_sim]


def preprocess(q):
    
    q = str(q).lower().strip()
    
    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    
    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')
    
    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    
    return q 

if __name__ == '__main__':
    dataset = sys.argv[1]
    percentage = float(sys.argv[2])

    data = pd.read_csv(dataset)

    # Number of rows to select 
    num_rows = int(len(data) * percentage)

    # Generate a list of random indices
    random_indices = random.sample(range(0, len(data)), num_rows)

    # Select the rows with the corresponding indices
    random_rows = data.iloc[random_indices]

    # Create a new DataFrame with the selected rows
    data_set = pd.DataFrame(random_rows)
    data_set = data.dropna()
    print('something')
    

    data_set['question1'] = data_set['question1'].apply(preprocess)
    data_set['question2'] = data_set['question2'].apply(preprocess)

    ngram_features = data_set.apply(ngram_features, axis=1)
    data_visualization(data_set)
    data_set = data_engineering(data_set)
    data_analysis(data_set)

    data_set.to_csv('dataset.csv')

