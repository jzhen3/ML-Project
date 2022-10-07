# CS7641-Proposal
Proposal for CS7641 project
____________________________________________________________________________________________________________________________________
https://www.kaggle.com/datasets/shivamb/netflix-shows  
https://www.kaggle.com/datasets/shivamb/amazon-prime-movies-and-tv-shows   
https://www.kaggle.com/datasets/shivamb/disney-movies-and-tv-shows   
https://www.kaggle.com/datasets/shivamb/hulu-movies-and-tv-shows  
____________________________________________________________________________________________________________________________________
Introduction/Background:
A majority of us tend to like watching programming content with similar features produced by the similar directors or casts. Since Bo Pang et al., in 2002, introduced the concept of emotional polarity classification, sentiment analysis on online reviews become more popular. It investigates subject texts involving users’ preferences and sentiment. Our movies and TV shows datasets are featured with originate variables like title, cast, director, ratings, release year, country of production, etc. and extracted from Netflix, Amazon Prime Video, Disney Plus, and Hulu.  

Problem Definition:
With rising demand of online subscription in online movies and TV shows, streaming services should better their recommendation systems. To retain their customers, streaming services usually release a wide variety of contents that meet the users’ satisfactions. Netflix started a science fiction horror drama television series Stranger Things with excellent production team since 2016. However, each season is released once in a while. A cancellation of subscription can result from the fact that when a user can’t find any other interesting to watch. Our objective is to reduce current challenges and apply popular machine learning algorithms to recommend tailored lists of movies or shows based on the common features of the users’ liked contents.  

Jinsong's References:
https://www.sigmamagic.com/blogs/netflix-data-analysis/#:~:text=One%2Dway%20Netflix%20uses%20to,end%20up%20cancelling%20the%20subscription.

https://www.sciencedirect.com/science/article/pii/S1877050922001107

## Methods, Matrixes and Potential Results Discussion
### Supervised Method:
1. Regression
2. Tress-based method 

#### Matrix for Supervised Methods: 
We choose several matrixes to evaluate models’ performance. Firstly, we will compute the confusion matrix and the calculate precision and recall rate as well as graphing of ROC AUC. Secondly, we will use some additional matrix to assist to evaluate models, such as the F-beta score

#### Potential Results Discussion for Supervised Methods:
Our group expect models will higher recall rate (0.7) because we think False negative, which means the recommendation system does not predict what clients like to see. If the model can achieve higher recall rate, we can accept relatively lower precision rate (0.6). Secondly, we expect a ROC graph’s angle close to the right angle and the AUC is greater than 0.7. F1 score consider both false positive and false negatives into account but we want to use F-beta score adjusting beta can show how much think recall rate is more important than precision rate. For example, if we set beta = 2, it means that the recall is twice as important as precision. 

### Unsupervised Methods:
1. K-means
2. Hierarchical Clustering Algorithm

#### Matrix for Unsupervised Methods: 
we will chose four matrix to evaluate models' performance
1.	fowlkes_mallows_score
2.	homogeneity_score
3.	mutual_info_score
4.	rand_score

#### Potential Results Discussion for Unsupervised Methods:
Our group expects models can generate a similar result with the testing set, so the homogeneity score and rand score should both be higher. And mutual info score should be relatively lower since clusters are expected to have high purity. Also, we expect Fowlkes mallows to score close to 1, which shows our prediction results are similar to the testing set.
