class Preprocessor:
    def __init__(self, rawData):
        self.rawData= rawData
    
    def normalize(self):

        temp= self.rawData.copy()
        temp= temp.str.lower()
        temp= temp.str.replace('[^\w\s]','')
        #convert strings to numbers
        
        self.processedData= temp
    
    def tokenization(self):
        pass

    def lemmatization():
        pass