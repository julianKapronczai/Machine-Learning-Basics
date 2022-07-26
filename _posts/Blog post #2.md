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