---
layout: post
title: Introduction to Machine Learning
---
### Machine Learning Introduction

<video src="https://user-images.githubusercontent.com/90798172/174352084-b0aebb3d-9935-4d29-8fc7-3d7c439012fb.mp4" data-canonical-src="https://user-images.githubusercontent.com/90798172/174352084-b0aebb3d-9935-4d29-8fc7-3d7c439012fb.mp4" controls="controls" muted="muted"  style="width: 100%; height: auto;">

  </video>

Machine learning is widely used in modern society. It allows programs and machines to learn depending on how the programmer implements them, and the data they’re presented with.  Everything from which advertisements you see on YouTube, to recommended friends on social media, are all the results of implemented machine learning algorithms. Other than the use of machine learning in the medias, there are more important application of machine learning in our society. Personalizing treatment plans, and analyzing micro bacteria are both examples of problems and tasks that can be solved by implementing machine learning.

Machine learning is in essence a machines ability to learn from mathematical data in a way that would otherwise be impossible for a human to replicate. We use what is called a feature matrix a large set of data, and a target vector the value you want output after the data is run through the program. Machine learning algorithms take data from a feature matrix and runs it through a model. A model is program that uses a mathematical algorithm to predict a target vector. This model can learn and be altered making the outputs of the program more accurate to the target vector data.  Now that all sounds simple, you have data, and you use machine learning models to produce an output that closely matches the target vectors. 

There are two different types of machine learning we will go over in this blog. Each type returns a different kind of target vector. The data you are presented with dictates which kind of machine learning algorithm are required, or what steps you will need to take to improve the algorithm’s performance. As the programmer you will provide a program with different types of algorithms for different data related tasks. As such, the data being used plays the biggest role in how machine learning will be implemented within a system. Down below you will see the two types please click on which you would like to start with. The recommended order is Supervised learning first, then moving on to Unsupervised learning. The figure bellow will help illustrate what we will be working with going forward (Figure #1).



![Figure1]({{site.url}}/assets/images/pic-1-for-blog.png)
(Figure 1)

### Supervised Learning

Supervised learning is the most commonly used type of machine learning. Typically, the types of problems you will want to use supervised learning for are problems where you know as much as possible about the inputs, and have some preexisting data for the desired outputs.  Supervised learning is implemented as a solution for both classification (Figure #2), and regression (Figure #3) problems. 


![Figure2]({{site.url}}/assets/images/pic-3-for-blog.png)
(Figure 2)
 ![Figure3]({{site.url}}/assets/images/pic-2-for-blog.png)
 (Figure 3)



As an example, let’s say you have three types of apples. Granny Smith, Red Delicious, and Gala. If we want to classify the different apples and predict the shelf life of each kind of apple, we have both a classification and a regression problem. Classification problems try to sort the samples into predefined, discrete categories, whereas regression problems try to use the data to predict a numerical value in a continuous range. To simplify that, classification predicts which class a data point should belong too, and associates it to a data point. Regression predicts a float point within a continuous range that depicts the relationship between data points. To use machine learning to analyze our apples, we will need to build a separate model for each problem. Our first problem we will deal with is the classification problem. We need our model to predict which apple belongs in which class of apple. Second, we will need to predict a continuous number that will give us an idea of how long each apple will survive on a shelf, making this the regression portion of the problem. For both problems, to start things off, we can split our data set in two parts: one set containing our training dataset, and the other being our test dataset. We will then be once again splitting our training dataset in two. This allows us to train our models with two different sets of training data in a process called cross validation. Cross validation allows us to improve our models by teaching with one half of the data to get our baseline accuracy. We then  evaluate the model with data the model hasn’t seen, and make adjustments to make our models as accurate as we can get them before exposing them to real world data. This is done repeatedly while adjusting the model during both training steps so we can get our predictions as accurate as possible to the target vectors. If, for whatever reason, our model  predicts with perfect accuracy, it suggests we likely we have made an error, as it will be nearly impossible to attain that accuracy once we introduce real data to our models. With that said, we now have cross validated our models and they ideally have a high accuracy for both the classification of apple and the regression of their shelf life. We are now ready to test our models on our test dataset, and the result of that will be the value that we advertise as the accuracy of our models on real world data.

<video src="https://user-images.githubusercontent.com/90798172/174354625-bf5d2972-e827-431a-a063-7ed11c51f2ab.mp4" data-canonical-src="https://user-images.githubusercontent.com/90798172/174354625-bf5d2972-e827-431a-a063-7ed11c51f2ab.mp4" controls="controls" muted="muted"  style="width: 100%; height: auto;">

  </video>


Now that you understand a bit more about how we can work with supervised learning read and watch below for a demonstration in how its implemented.


### Unsupervised Learning


Unsupervised learning is designed to make enormous datasets easier to digest and utilize by either dimensionality reduction or clustering. Unsupervised learning algorithms don’t come without their share of limitations. Although it’s great for saving time and making data easier to visualize, it is incredibly hard to be sure your algorithm learns anything useful. Unless you are interested in combing through massive datasets, then it’s better to just use them when applicable. These algorithms are used primarily for data representation by pre-processing or rescaling the data. Transforming data using unsupervised learning reduces the complexity, and represents data in a labeled or sorted format, which allows us to further process the data if need be. Generating these labeled or sorted datasets take large amounts of manpower, and is often outsourced to third world countries. This is where supervised learning shines in its efficiency by reducing the required man hours; however it can fall short. The tradeoff in some cases for the labor cost of real humans, versus inaccuracies that show up in machine learning predictions (or poorly paid humans), are all things that need to be considered. Regardless the implementation of unsupervised learning algorithms will reduce the impact the dataset has on our results, and the man hours required to sort through datasets.


One method of unsupervised learning is clustering the data. Clustering is a process that partitions a dataset into groups called clusters. It sections off the data points of similar nature with other similar data points, somewhat like the classification of datasets talked about in supervised learning. The clustering algorithm assigns a number to each data point designating it to a specific cluster. The easiest example of clustering to understand is k-means clustering. K-means clustering attempts to locate the middle of a cluster to represent each section of data. The algorithm switches between assigning each data point to the most relevant cluster based on its features and reassigning each cluster’s center point based on the mean of all the newly assigned data points. Once the algorithm runs its course, the cluster instances will no longer be changed. The image below helps to illustrate that (Figure #4).



![Figure5]({{site.url}}/assets/images/pic-5-for-blog.png)
(Figure 5)

<video src="https://user-images.githubusercontent.com/90798172/174352716-1d82d5e4-ed3d-4033-b0de-105afb0426d7.mp4" data-canonical-src="https://user-images.githubusercontent.com/90798172/174352716-1d82d5e4-ed3d-4033-b0de-105afb0426d7.mp4" controls="controls" muted="muted" style="width: 100%; height: auto;">

  </video>



<sub>Reference List:

1. Andreas C. Müller, Sarah Guido - Introduction to Machine Learning with Python_ A Guide for Data Scientists-O’Reilly Media (2016)
2. David Cooksley, Human Resource, Mentor
3. P. Bozelos, “World wide ✈️ machine learning,” World Wide ML. [Online]. Available: https://www.world-wide.org/ML/. [Accessed: 17-Jun-2022]. 
4. “Supervised and unsupervised machine learning - explained through real world examples,” Omdena, 21-Mar-2022. [Online]. Available: https://omdena.com/blog/supervised-and-unsupervised-machine-learning/. [Accessed: 17-Jun-2022]. 
5. “Clustering in machine learning,” GeeksforGeeks, 18-May-2022. [Online]. Available: https://www.geeksforgeeks.org/clustering-in-machine-learning/. [Accessed: 17-Jun-2022].  </sub>

