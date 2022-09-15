import string
import numpy as np
import pandas as pd

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.notebook import tqdm

# Stemming seems to cause lots of inconsistency issues when it comes to vectorising the stop words. 
# Should look into this further as I think it could help accuracy. But from what I've read the main goal of stemming is
# is speed which is not a concern here.
def stem_tokens(tokens):
    """Stems a list of tokens using a modified version of the Porter stemming algorithm
    provided by the nltk package.
    
    Parameters
    ----------
    tokens: list
        a list of tokens
        
    Returns
    ---------
    stemmed_tokens: list
        a list of stemmed tokens
    """
    # Load in Martin Porter's stemming algorithm
    stemmer = PorterStemmer()
    
    # Stem each token in the input list
    stemmed_tokens = [stemmer.stem(item) for item in tokens]
    return stemmed_tokens

def lemmatise_tokens(tokens, pos):
    """Lemmatises a list of tokens using the word net lemmatiser provided by the nltk package.
    
    Parameters
    ----------
    tokens: list
        a list of tokens
    
    pos: string
        one of 'n' for nouns, 'v' for verbs, 'a' for adjectives, 'r' for adverbs,
        's' for satellite adjectives
        
    Returns
    ---------
    lemmatised_tokens: list
        a list of lemmatised tokens
    """
    # Lemmatise the tokens using the word net lemmatiser
    lemmatised_tokens = [WordNetLemmatizer().lemmatize(word, pos = pos) for word in tokens]
    
    return lemmatised_tokens
    
def normalise(text):
    """Normalises text data tokenising, removing punctation, setting everything to lower case
    and finally lemmatising.
    
    Punctation list provided by the string module.
    
    Parameters
    ----------
    text: string
        a text document
        
    Returns
    ---------
    tokens: list
        a list of normalised tokens
    """
    # First we tokenise
    tokens = word_tokenize(text)    
    
    # Define a map which removes punctation
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

    # Replace the punctation with an empty string
    puncs = [word.translate(remove_punctuation_map) for word in tokens]
    
    # Remove the empty strings
    puncs_removed = list(filter(None, puncs))
                         
    # Then set everything to lower case
    lower = [word.lower() for word in puncs_removed]
    
    # Finally we lemmatise everything
    lemmatised_tokens = lemmatise_tokens(lower, pos = 'v')
    
    return lemmatised_tokens

# Make sure we normalise the stop words according to how we are normalising the text
# to ensure consistency.
stopwords = [normalise(word)[0] for word in stopwords.words('english')]

def cosine_sim(doc_1, doc_2):
    """Calculate the cosine similarity between two text documents.
    
    Parameters
    ----------
    doc_1: string
        the first document
    
    doc_2: string
        the second document
        
    Returns
    ---------
    cos_sim: float between 0 and 1
        the cosine similarity between document 1 and 2
    """
    # Define a vectoriser using the term frequency-inverse document frequency (TF-IDF) measure
    # (The more frequent a word appears, the less important it is)
    vectoriser = TfidfVectorizer(tokenizer = normalise, stop_words = stopwords)
    
    # Learn the vocabulary of the two documents, and transform to vectors using TF-IDF.
    # This produces normalises vectors so no need to normalise when computing cosine similarity.
    tf_idf = vectoriser.fit_transform([doc_1, doc_2])
    
    # Calculate the pairwise similarity matrix
    pairwise_similarity = ( tf_idf * tf_idf.T ).toarray()
    
    # Return the similarity
    return pairwise_similarity[0, 1]

def similarity_subset(input_data_path, output_data_path, threshold = 0.8, return_ = False):
    """Function to load data consisting of two columns, the first containing old company blurbs
    the second containing new blurbs, work out their similarity and return a new data file consisting 
    of only those blurbs that are more different than some user-specified threshold.
    
    Parameters
    ----------
    input_data_path: string
        the path to the data file, the data must be an excel table
    
    output_data_path: string
        the path where you want save the new data file e.g. './new_data.ods'
    
    threshold: float betweeon 0 and 1
        the cut off point at which we filter out a blurb for not being similar enough,
        the higher the threshold, the more similar we allow blurbs to be
        
    return_: boolean
        decides whether or not the new dataset is returned
    """
    # Read in data, drop the empty rows
    data = pd.read_excel(input_data_path, engine = 'odf').dropna().to_numpy()
    
    N = data.shape[0]
    print(f'Number of blurbs: {N}')
    
    # Remove those rows that are identical
    data = data[data[:, 0] != data[:, 1]]
    print(f'Removed {N - data.shape[0]} identical blurbs...')
    
    # Extract remaining blurbs
    old_blurbs = data[:, 0].tolist()
    new_blurbs = data[:, 1].tolist()
    
    # Compute similarities and remove the rows that are below the threshold
    print('Computing similarities of remaining blurbs...')
    similarities = np.zeros(len(old_blurbs))
    for i in tqdm(range(len(old_blurbs))):
        similarities[i] = cosine_sim(old_blurbs[i], new_blurbs[i])
    
    M = sum(similarities < threshold)
    print(f'Remaining number of blurbs: {M}. Total reduction of {round((N - M)/N * 100, 2)}%.')
    
    print(f'Saving file to the path {output_data_path}')
    # Find those blurbs where the similarity is less than our threshold, 
    # and output them to a new excel file
    threshed_indexes = np.where(similarities < threshold)
    new_data = pd.DataFrame( data[threshed_indexes] )
    new_data.to_excel(output_data_path,
                      index = False,
                      header = ['Old - Own blurb from the bridge',
                                'New - Own blurb you would copy paste onto the bridge'])
    
    # If we wanted to return the new dataset, return it
    if return_:
        return new_data
