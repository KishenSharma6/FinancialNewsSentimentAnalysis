import nltk

class Preprocessor:
    def __init__(self, rawData):
        self.rawData= rawData
    
    def normalization(self):
        temp= self.rawData.copy()
        temp= temp.str.lower()
        temp= temp.str.replace('[^\w\s]','', regex= True)
        self.normalizedData= temp
    
    def tokenization(self):
        tokens= self.normalizedData.str.split()
        self.tokens= tokens
   
    def lemmatization(self):
        self.lemmaTokens= self.tokens.apply(lemmatize_text)

class DataRepresentation:
    pass


def lemmatize_text(text):
    lemmatizer= nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text]