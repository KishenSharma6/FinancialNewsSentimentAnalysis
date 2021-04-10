import nltk
import sklearn

def lemmatize_text(text):
    """Function lemmatizes input text. This function is to be used in the Preprocessor class
    """
    lemmatizer= nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text]


class Preprocessor:
    def __init__(self, rawData):
        self.rawData= rawData
    
    def normalization(self):
        """Normalizes self.rawData. Puncutation is removed and all words are converted to lowercase
        """
        temp= self.rawData.copy()
        temp= temp.str.lower()
        temp= temp.str.replace('[^\w\s]','', regex= True)
        self.normalizedData= temp
    
    def tokenization(self):
        """Removes stopwords fron self.nomalizedData and converts strings into tokens
        """
        from nltk.corpus import stopwords
        stopWords= set(stopwords.words('english'))
        tokens= self.normalizedData.str.split()
        filteredTokens= tokens.apply(lambda x: [token for token in x if token not in stopWords])
        self.tokens= filteredTokens
   
    def lemmatization(self):
        """Applies lemmatization to to self.tokens and appends to self as self.lemmaTokens
        """
        def lemmatize_text(text):
            lemmatizer= nltk.stem.WordNetLemmatizer()
            return [lemmatizer.lemmatize(w) for w in text]
        self.lemmaTokens= self.tokens.apply(lemmatize_text)

class DataRepresentation:
    def __init__(self, text_data, target):
        self.text= text_data
        self.target= target

    def split_data(self, trainingSize= .8, testSize= .2, stratify=None):
        """Splits self.text and self.target into 

        Args:
            trainingSize (float, optional): [description]. Defaults to .8.
            testSize (float, optional): [description]. Defaults to .2.
            stratify ([type], optional): [description]. Defaults to None.
        """
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test= train_test_split(self.text, self.target,
                                                           train_size= trainingSize, test_size=trainingSize, stratify=stratify, random_state= 24)
        self.X_train= X_train
        self.y_train= y_train
        self.X_test= X_test
        self.y_test= y_test

    def bag_of_words(self):
        """Create BoW representation of object's training data

        Returns:
            sparse matrix: Returns BoW of self.X_train
        """
        from sklearn.feature_extraction.text import CountVectorizer
        vect= CountVectorizer().fit(self.X_train)
        X_train_BoW= vect.transform(self.X_train)
        return X_train_BoW


