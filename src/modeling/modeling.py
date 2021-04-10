import sklearn
import numpy as np

class Models:
    def __init__(self, training_data, target):
        self.training= training_data
        self.target= target
    
    
    
    def base_model_evaluation(self):
        #Initilialize Base Models
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import LinearSVC
        
        logR= LogisticRegression(multi_class= "multinomial", n_jobs= -1)
        randomF= RandomForestClassifier(n_jobs= -1, random_state= 24)
        knn= KNeighborsClassifier(n_jobs= -1)
        linearSVC= LinearSVC(multi_class= "crammer_singer", random_state= 24)
        
        from sklearn.model_selection import cross_val_score
        baseModels= [logR, randomF, knn, linearSVC]
        for model in baseModels:
            scores= cross_val_score(model, self.training, self.target, cv=5)
            print("=========================")
            print("%s Scores: %s" % (model, scores))
            print("Average Sccore: %s" % (round(np.mean(scores),2)))
            print("Standard Deviations: %s" % (round(np.std(scores),2)))