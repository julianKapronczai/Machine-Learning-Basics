---
layout: post
title: Introduction to Machine Learning
---
### Machine Learning Introduction
Machine learning is widely used in today’s society. It allows programs and machines to learn depending on how the programmer implements them, and the data its presented with.  Everything from advertisements you see on YouTube, to recommended friends on social media, are all the results of implemented machine learning algorithms. Other than the use of machine learning in the medias, there are more important  application of machine learning in our society. Things like personalizing treatment plans, and analyzing micro bacteria are examples of problems and tasks that can be solved by implementing machine learning.

Machine learning is in essence a machines ability to learn from mathematical data that would otherwise be impossible for a human to reproduce. We use what is called a feature matrix a large set of data, and a target vector the value you want output after the data is run through the program. Machine learning algorithms take data from a feature matrix and runs it through a model. A model is program that uses a mathematical algorithm to predict a target vector. This model can learn and be altered making the outputs of the program more accurate to the target vector data.  Now that all sounds simple, you have data, and you use machine models to produce an output that closely matches the target vectors. 

There are two different types of learning we will go over in this blog. Each type of machine learning returns different kind of target vectors. The data you are presented with dictates which kind of machine learning algorithm is required, or what steps with multiple algorithms you will need to take. As the programmer you will provide a program with different algorithms for different data related tasks. As such the data being used plays the biggest role in how machine learning will be implemented within a system. Down below you will see the three types please click on which you would like to start with. The recommended order is from Supervised learning moving on to Unsupervised learning. The figure bellow will help illustrate what we will be working with going forward(Figure #1).


![Figure1]({{site.url}}/assets/images/pic-1-for-blog.png)

### Supervised Learning

Supervised learning is more commonly used than other types of machine learning. Most often the types of problems you will want to use supervised learning for are problem where you have preexisting data for the inputs, as well as the outputs.  Supervised learning is implemented as   a solution for classification(Figure #2), and regression problems(Figure #3). 


![Figure2]({{site.url}}/assets/images/pic-2-for-blog.png) ![Figure3]({{site.url}}/assets/images/pic-3-for-blog.png)



So, let’s say you have three fruits. Granny Smith apples, Red Delicious apples, and Gala apples, all these fruits have lots of distinguishing traits that make them unique. This means we have either a classification problem or a regression problem. We want to classify the different apples, and evaluate the shelf life of each kind of apple. For both we will need to build a model according to these target vectors. Our first problem we will deal with is the classification problem, we need out model to predict which apple belongs in which class of apple. Second we will need to predict a continuous number that will give us an idea of how long each apple will survive on a shelf, making this the regression portion of the problem. To start things off we can split our data set in half, one set containing our training dataset and another being our test dataset. We will the be once again splitting our training dataset in half again, this allows us to train our models with two different sets of training data. Doing so is called cross validation. Cross validation allows us to improve our models by teaching with one half of the data to get our baseline accuracy. Then teach with data the model hasn’t seen improving the models as we go again to have our models as accurate as we can get them before exposing them to real world data. This is done repeatedly while adjusting the model during both teaching steps so we can get our predictions as accurate as possible to the target vectors. If for what ever reason we are getting our model to predict with perfect accuracy its shows we likely we have made an error as it will be nearly impossible to attain that accuracy once we introduce real data to our models. With that said we now have cross validated out models and they ideally have a high accuracy for both the classification of apple and the regression of their shelf life. We are now ready to test out data on our test dataset and the result of that will be the final result of the accuracy of our models on real world data. 

Now that you understand a bit more about how we can work with supervised learning read and watch below for a demonstration in how its implemented. 


### Unsupervised Learning


Unsupervised learning is designed to make enormous datasets easier to digest and utilize by either dimensionality reduction or clustering. Unsupervised learning algorithms don’t come without their share of limitations, all though its great for saving time and making data easier to see, it is incredibly hard to be sure your algorithm learns anything useful. Unless you are interested in combing through massive dataset then its better to just use them when applicable. These algorithms are used primarily for data representation by pre-processing or rescaling the data. Transforming data using unsupervised learning compresses, and represents data in a more informative way, which allows us to further process the data if need be.  This will in turn reduce the impact the dataset has on our results.


One way of doing so is by clustering the data. Clustering is the rescaling process that partitions a dataset into groups called clusters. Its sections off the data points of similar nature to other data points similar to the classification of datasets talked about in supervised learning. The clustering algorithm assigns a number to each data point designating it to a specific cluster. The easiest example of clustering to understand is k-means clustering. K-means clustering attempts to locate the middle of a cluster to represent each section of data. The algorithms switches between assigning each data point to the most relevant cluster based on its features. Then assigning each cluster center point based on the mean of all the newly assigned data points. Once the algorithm runs its course the cluster instances will no longer be changed. The image bellow helps to illustrate that(Figure #4).  


![Figure5]({{site.url}}/assets/images/pic-5-for-blog.png)




<sub> Huge Thanks to Alex Hill for being my best friend </sub>

