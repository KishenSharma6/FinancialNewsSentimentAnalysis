import sklearn
import numpy as np

class Models:
    def __init__(self, training_data, target):
        self.training= training_data
        self.target= target
   
    def base_model_evaluation(self):
        #Initilialize Base Models
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import LinearSVC
        
        logR= LogisticRegression(multi_class= "multinomial", n_jobs= -1, random_state= 24)
        mnb= MultinomialNB()
        randomF= RandomForestClassifier(n_jobs= -1, random_state= 24)
        knn= KNeighborsClassifier(n_jobs= -1)
        linearSVC= LinearSVC(multi_class= "crammer_singer", random_state= 24)
        
        from sklearn.model_selection import cross_validate
        baseModels= [logR, mnb, randomF, knn, linearSVC]
        names= ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'KNN', 'Linear SVC']
        scoring = {'acc': 'accuracy',
                   'f1_macro': 'f1_macro'}
        for name, model in zip(names, baseModels):
            scores= cross_validate(model, self.training, self.target, 
                                    scoring= scoring, cv=5)
            print("=========================")
            print("%s Base Performance Metrics:" % (name))
            print("Average Accuracy: %s" % (round(np.mean(scores['test_acc']),2)))
            print("Accuracy Standard Deviation: %s" % (round(np.std(scores['test_acc']),2)))
            print("Average F1 Macro: %s" % (round(np.mean(scores['test_f1_macro']),2)))
            print("Accuracy F1 Macro Standard Deviation: %s" % (round(np.std(scores['test_f1_macro']),2)))


    def random_search_cv(self, estimator, parameters):
        """
        Apply RandomSearchCV and return tuned model
        """
        from sklearn.model_selection import RandomizedSearchCV
        clf= RandomizedSearchCV(estimator, param_distributions= parameters,
                                random_state= 24, cv= 5)
        search= clf.fit(self.training, self.target)
        return search   

    def get_predictions(self, model):
        """Returns predictions from model
        """
        predictions= model.predict(self.training)
        return predictions