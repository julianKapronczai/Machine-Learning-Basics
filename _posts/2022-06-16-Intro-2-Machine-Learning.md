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


### Implementing Machine learning

First we will need an environment to work in, for this we will need to install Anaconda3. This will allow us to work in Jupyter notebook using Numpy, Pandas, Scikit-learn, and Matplotlib. Use the link bellow and follow along once it downloaded.

https://www.anaconda.com/products/distribution 
 ![Figure6]({{site.url}}/assets/images/anadown1.png)

Next up run the installer and follow along with the instructions. 

Click: Next

 ![Figure7]({{site.url}}/assets/images/anadown2.png)
Click: I Agree

 ![Figure8]({{site.url}}/assets/images/anadown3.png)
Install for: Just Me 
Then click: Next

 ![Figure9]({{site.url}}/assets/images/anadown4.png)
Select your desired destination folder then click: Next

 ![Figure10]({{site.url}}/assets/images/anadown5.png)
Leave the two boxes un-checked and click: Install

 ![Figure11]({{site.url}}/assets/images/anadown6.png)


The goal of this portion of the blog is to get a better understanding of how machine learning is implemented. To do so I will run you through some very simple examples of the tools we will use before we learn about implementing machine learning models properly.

Please follow the steps bellow and we will be good to get started with the examples:

1.Open up the Anaconda Navigator:
 
 ![Figure12]({{site.url}}/assets/images/anarun1.png)
2.Then launch JupyterLab:

 ![Figure13]({{site.url}}/assets/images/anarunJup.png)
3.Your browser will open this window:

 ![Figure14]({{site.url}}/assets/images/anarun2.png)
4.You now want to click this blue '+' button:

 ![Figure15]({{site.url}}/assets/images/newanaproj.png)
5.Click the Python 3 button under the Notebook section:

 ![Figure16]({{site.url}}/assets/images/newanaproj2.png)
You are now ready to start coding in Python 3, as well as work with machine learning.


REALLY IMPORTANT NOTE!!!!
To run the code blocks you write you will need to press the 'shift' key + the 'enter' key.

Now im not going to teach you Python as a language but really qyuick im going to mention some of the tools we will be using for the coding examples. To start off all of our examples we are going to be importing Numpy, Pandas, matplotlib, and seaborn


Numpy is a very useful package containing n-dimensional arrays and tool to integrate c and c++. Pandas provides us with a package to analyse and manipulate data structures. Matplotlib provides us with a library that allows us to visualize the data we will be working with. Lastly we will be using Seaborn which is another data visualization library, its bassed on matplot and provides further ways to visualize the data. 

You will want to start off almost any project by importing these libraries as they are core performers for anyone working with machine learning.

Your import code should look like this:
![Figure17]({{site.url}}/assets/images/imp1.png)


The first example of implementation wil be a regression example. The goal of this is to become more familiar with regression models. We will be using some sample data about concrete mixtures to try and predict the strength of different concrete mixtures using a regression model.

Make sure the libraries above are imported in you code. 

We will now deffine the functions we will be using. Root-mean squared error (rms) will be used as our scoring function. An optimal score for rms is 0, the larger the number gets the worse the performance of a model. With that said in scikit-learn we can only use maximization. so we will mave to maximize the negative rms, scikit-learn does provide a scoring function we can use here neg_root_mean_squared_error. We will also calculate the score for the model using cross-validation this score will be the rms.

Here is the code you will need. 
![Figure18]({{site.url}}/assets/images/imp2.png)

Now we need to load the data we will be working with. The concrete data you can find at this link: https://www.scikit-yb.org/en/latest/api/datasets/concrete.html
To get more info on this data set we will need to print out the README of the concrete dataset. To do so we need to load the dataset object using return_dataset=True. 
![Figure19]({{site.url}}/assets/images/imp3.png)

Now that we have information on the dataset, we can now load the concrete dataset into feature matrix 'X' and target vector 'y'
![Figure20]({{site.url}}/assets/images/imp4.png)
We now can see what our data actually looks like so lets print out the X.min() and X.max()
![Figure21]({{site.url}}/assets/images/imp5.png)
As nice as that is we can use seaborn to visualize this data a little bit better. Using the code bellow you can print out a heatmap of all of features in our dataframe
![Figure22]({{site.url}}/assets/images/imp6.png)
Now we need to create our traning and testing datasets. To do that we will use scikit-learn train_test_split() with the parameters random_state=37, test_size=0.2, split X and y into training and test sets.
![Figure23]({{site.url}}/assets/images/imp7.png)
Comparing models with cross-validation allows us to establish which model will be better for our usecase. We are going to use 2 models for this LinearRegression,  RandomForestRegressor(random_state64) and GradientBoostingRegressor(random_state79). We will use seven fold cross-validation 

To iterate this list we need to get the negative root mean squared error. We can get the negRMS with get_regressor_neg_rms() function and pring out the results to 2 deciman places.
![Figure24]({{site.url}}/assets/images/imp8.png)

All of those models had some problems with over and underfitting as shown by the printed results. Which means we need to find a better model. 

To do so we will use the RandomForestRegressor(random_state=64) again but find a better combination of max_depth from a list of [10, 15, 20] and n_estimators from a list of [50, 100, 150] 

Again we will use the get_regressor_neg_rms() function to pring out our results to 2 decimal places. 
![Figure20]({{site.url}}/assets/images/imp9.png)

As we can see the highest validation score of -4.93 came from max_depth=20 and n_estimators=150 

With our model decided on we now will retrain the superior model. We will create a new RandomForestRegressor(random_state=64) with the best max_pair and n_estimators, then train it on all of our training data. 
![Figure20]({{site.url}}/assets/images/imp10.png)

Now that our model is trained we can evaluate how it performs and see what our root mean-squared error(rmse) result is.
![Figure20]({{site.url}}/assets/images/imp11.png)
As we can see our training rmse is 0.986 and our testing rmse is 0.913 

Lets get a better visualization of what that looks like using a scatterplot that shows predicted strength on the x axis, and actual strength on the y axis. That way we can see where our errors were made.

![Figure20]({{site.url}}/assets/images/imp12.png)


The regression example above is a good introduction into the machine learning process. First we create our training and test splits, then we pick out our model using cross validation, we then receive a final score on the initial split. Also throughout the process we played around with some data visialization to further our understanding of what we are working with. 

Next up is a classification problem. This time using data about poisionous and non-poisonous mushrooms. 





<sub>Reference List:

1. Andreas C. Müller, Sarah Guido - Introduction to Machine Learning with Python_ A Guide for Data Scientists-O’Reilly Media (2016)
2. David Cooksley, Human Resource, Mentor
3. P. Bozelos, “World wide ✈️ machine learning,” World Wide ML. [Online]. Available: https://www.world-wide.org/ML/. [Accessed: 17-Jun-2022]. 
4. “Supervised and unsupervised machine learning - explained through real world examples,” Omdena, 21-Mar-2022. [Online]. Available: https://omdena.com/blog/supervised-and-unsupervised-machine-learning/. [Accessed: 17-Jun-2022]. 
5. “Clustering in machine learning,” GeeksforGeeks, 18-May-2022. [Online]. Available: https://www.geeksforgeeks.org/clustering-in-machine-learning/. [Accessed: 17-Jun-2022].  
6. Yeh, I-C. "Modeling of strength of high-performance concrete using artificial neural networks." Cement and Concrete research 28.12 (1998): 1797-1808.
</sub>

