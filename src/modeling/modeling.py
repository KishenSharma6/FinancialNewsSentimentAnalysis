import sklearn

class Models:
    def __init__(self, training_data, target):
        self.training= training_data
        self.target= target
    

    #Initialize base models
    
    
    def base_model_evaluation(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import LinearSVC
        
        logR= LogisticRegression(multi_class= "multinomial", n_jobs= -1)
        randomF= RandomForestClassifier(n_jobs= -1, random_state= 24)
        knn= KNeighborsClassifier(n_jobs= -1)
        linearSVC= LinearSVC(multi_class= "crammer_singer", random_state= 24)
        
        baseModels= [ logR, randomF, knn, linearSVC]
        for model in baseModels:
            print(model)