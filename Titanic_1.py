"""
@author Maya Allalouf
@Titanic survivals dataset analysis
In order to Find attributes that are the main drivers for survival I will go through couple of steps:
1. Data engineering
2. Data visualization by graphs.
3. Machine learning models: Logistic Regression.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import scipy.stats as stats
import seaborn as sns
import re
from sklearn.linear_model import LogisticRegression

class Titanic: 

    def plotMultDist(self,first_dist, second_dist, title):
        '''
        Plot a histogtam for a continuous feature for two different distributions. 
        
        :param first_dist: distribution to plot
        :param second_dist: distribution to plot
        :param title: The graph title
        ''' 
        fit_survivals = stats.norm.pdf(first_dist, np.mean(first_dist), np.std(first_dist))  
        fit_not_survivals = stats.norm.pdf(second_dist, np.mean(second_dist), np.std(second_dist))  
        sns.set(color_codes=True)
        sns.distplot(first_dist, label='Survivals')
        sns.distplot(second_dist, label='Not Survivals')
        plt.title(title)
        plt.legend()
        plt.show()
        return
    
    def barChart(self,first_distribution, second_distribution, binValues, featureName, SexKeys,EmbarkedKeys):
        '''
        Plot a barChart for a categorical feature for two different distributions
        
        :param first_dist: distribution to plot
        :param second_dist: distribution to plot
        :param binValues: The bins values for the x axis
        :param featureName: The graph title
        :param sexKeys: just for the Sex feature - The bins values for the x axis
        :param EmbarkedKeys: just for the Embarked feature - The bins values for the x axis
        ''' 
        survivalsFeaturesDist = np.zeros(0)
        notSurvivalsFeaturesDist = np.zeros(0)
        for index in binValues:
            count1 = np.count_nonzero(first_distribution==index)
            survivalsFeaturesDist = np.append(survivalsFeaturesDist,[count1])
            count2 = np.count_nonzero(second_distribution==index)
            notSurvivalsFeaturesDist = np.append(notSurvivalsFeaturesDist,[count2])
        width = 0.35       
        p1 = plt.bar(binValues, survivalsFeaturesDist, width)
        p2 = plt.bar(binValues, notSurvivalsFeaturesDist, width, bottom=survivalsFeaturesDist)
        #plt.title()
        if featureName == "Sex":
            plt.xticks(np.arange(2), SexKeys)
        if featureName == "Embarked":
            plt.xticks(np.arange(3),EmbarkedKeys)
        else:
            plt.xticks(binValues)    
        plt.xlabel(featureName)
        plt.legend((p1[0], p2[0]), ('survivals', 'not_survivals'))
        plt.show()
        return
        
    def devideAccordingLabels(self,df):
        '''
        Divide a dataframe into two different dataframes according to their labels (for binary labels)
        
        :param df: The dataframe to divide.
        :return: First dataframe includes all the first labels,  Second dataframe includes all the second labels
        '''
        positive_df = df['Survived'] == 1
        positive_df = df[positive_df]

        negative_df = df['Survived'] == 0
        negative_df = df[negative_df]
        
        return positive_df, negative_df
    
    def getFeatureValuesArray(self,df,feature):
        '''
        Get the values of a spesific feature (column) from a dataframe
        
        :param df: The dataframe 
        :param feature: the feature to get its values 
        
        :return: array of the feature values
        '''
        featureArray = np.array(df[feature])
        featureArray = featureArray.astype(int)
        return featureArray
        
    def logisticRegression(self,df,feature,label):
        '''
        Divide the dataframe into train set and test set. Train the classification model on the training set, by a spesific feature,
        predict the labels of the test set and print the Logistic regression mean accuracy.
         
        :param df: The dataframe 
        :param feature: the feature to train the model with.
        :param label: the label column in the dataframe
        '''
        
        msk = np.random.rand(len(df)) < 0.7
        train = df[msk]
        test = df[~msk]
        X = train[[feature]]
        X = (np.array(X).astype(float))
        y = train[label]
        y = (np.array(y).astype(float))
        clf = LogisticRegression().fit(X, y)
        #y = y.astype(int)
        Xtest = test[[feature]]
        Xtest = (np.array(X).astype(float))
        ytest = test[label]
        ytest = (np.array(y).astype(float))
        clf.predict(Xtest)
        print (str(feature)+" - Logistic regression mean accuracy: " + str(clf.score(Xtest, ytest)))
        return


 # Main execution       
                        
T = Titanic()
train_df = pd.read_csv('C:/Users/Maya/Documents/data_ex/titanic/train.csv')

# Part 1: Data engineering

# Map categorical features to numerical values and save the original values. 
train_df['Sex'] = train_df['Sex'].map({ 'male' : 0, 'female' : 1}) 
SexKeys = ('male','female')  
train_df['Embarked'] = train_df['Embarked'].map({ 'Q' : 0, 'C' : 1, 'S' : 2})
EmbarkedKeys = ('Q','C','S')

# Exrtact the title from the Name column, map it to numerical value and store the numerical value in a new column - 'Title'
train_df["Title"] = ""
for i in train_df.index:
    title = re.findall("\w*\.", train_df.at[i, 'Name'])
    if title[0] == 'Mr.':
        train_df.at[i, 'Title'] = 0
    elif title[0] == 'Mrs.':
        train_df.at[i, 'Title'] = 1
    elif title[0] == 'Miss.':
        train_df.at[i, 'Title'] = 2
    else:
        train_df.at[i, 'Title'] = 3
        
# Calculate the mean age for every title
Mr = train_df[(train_df['Title'] == 0)]
Mrs = train_df[(train_df['Title'] == 1)]
Miss = train_df[(train_df['Title'] == 2)]
Other = train_df[(train_df['Title'] == 3)]
meanAgeMr = (Mr['Age']).mean()
meanAgeMrs = (Mrs['Age']).mean()
meanAgeMiss = (Miss['Age']).mean()
meanAgeOther = (Other['Age']).mean()

# Define the missing 'Age' values as the mean age values of the person title
for i in train_df.index:
    if pd.isnull(train_df.at[i,'Age']):
        if train_df.at[i,'Title'] == 0:
            train_df.at[i,'Age'] = meanAgeMr
        elif train_df.at[i,'Title'] == 1:
            train_df.at[i,'Age'] = meanAgeMrs
        elif train_df.at[i,'Title'] == 2:
            train_df.at[i,'Age'] = meanAgeMiss
        else:
            train_df.at[i,'Age'] = meanAgeOther

# Create another feature that will sum the number of the relatives on the boat.            
train_df["RelativesNum"] = ""
train_df["RelativesNum"] = train_df["SibSp"] + train_df["Parch"]
        
# Drop features that seems less relevant 
train_df = train_df.drop(['Ticket', 'Name', 'Cabin'], axis = 1)

# Drop rows with na values (only two rows)
train_df = train_df.dropna()


# Part 2: Visualization and classification by feature:

# Divide the features to continuous and categorical features
categoricalFeatures = np.array(["Pclass","Sex","SibSp","Parch","Embarked","RelativesNum"]) 
continuousFeatures = np.array(["Age","Fare"])

# Divide the Data to survivals and not survivals 
survivalsTrain_df, notSurvivalsTrain_df = T.devideAccordingLabels(train_df)

# For every categorical feature:
#1. plot the values distribution for survivals and not survivals in a bar plot
#2. display the accuracy score of the logistic regression classification using the fature
for i in range (0,len(categoricalFeatures)):
    plt.figure()
    survivalsTrainFeature = T.getFeatureValuesArray(survivalsTrain_df,categoricalFeatures[i])
    notSurvivalsTrainFeature = T.getFeatureValuesArray(notSurvivalsTrain_df,categoricalFeatures[i])
    T.barChart(survivalsTrainFeature, notSurvivalsTrainFeature, np.unique(np.array(train_df[categoricalFeatures[i]])),categoricalFeatures[i],SexKeys,EmbarkedKeys)
    T.logisticRegression(train_df, categoricalFeatures[i],'Survived')

# For every continuous feature:
#1. plot the values distribution for survivals and not survivals in a histogram
#2. display the accuracy score of the logistic regression classification using the fature    
for i in range(0,len(continuousFeatures)):
    plt.figure()
    survivalsTrainFeature = T.getFeatureValuesArray(survivalsTrain_df,continuousFeatures[i])
    notSurvivalsTrainFeature = T.getFeatureValuesArray(notSurvivalsTrain_df,continuousFeatures[i])
    T.plotMultDist(survivalsTrainFeature, notSurvivalsTrainFeature, continuousFeatures[i])
    T.logisticRegression(train_df, continuousFeatures[i],'Survived')
    
# Check if the Sex (male/female) affects the correlation between Age feature and survival.

female_df = train_df['Sex'] == 1
female_df = train_df[female_df]

male_df = train_df['Sex'] == 0
male_df = train_df[male_df]

survivalsFemale_df, notSurvivalsFemale_df = T.devideAccordingLabels(female_df)
survivalsmale_df, notSurvivalsmale_df = T.devideAccordingLabels(male_df)

plt.figure()
survivalsTrainFeature = T.getFeatureValuesArray(survivalsFemale_df,'Age')
notSurvivalsTrainFeature = T.getFeatureValuesArray(notSurvivalsFemale_df,'Age')
T.plotMultDist(survivalsTrainFeature, notSurvivalsTrainFeature, 'Female Age')
T.logisticRegression(female_df, 'Age','Survived')

plt.figure()
survivalsTrainFeature = T.getFeatureValuesArray(survivalsmale_df,'Age')
notSurvivalsTrainFeature = T.getFeatureValuesArray(notSurvivalsmale_df,'Age')
T.plotMultDist(survivalsTrainFeature, notSurvivalsTrainFeature, 'Male Age')
T.logisticRegression(male_df, 'Age','Survived')


 

