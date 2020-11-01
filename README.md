# Collaborative filtering for Netflix movie ratings
This is a homework from the MIT's massive open online course (MOOC) [Machine Learning with Python](https://www.edx.org/course/machine-learning-with-python-from-linear-models-to). The idea was to build a mixture model almost from scratch to predict Netflix movie ratings from a sparse data matrix using a Gaussian distribution. 

Background for unsupervised learning concepts can be found quite easily by Googling or hopping straight into Wikipedia, but the author would suggest glancing the included lecture notes he took summarising the lecture material. The notes cover specifically information needed to understand the problem at hand. 

Run the code/main.py to see the results. The [RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation) was a little under 0.5, i.e. an OK performance when working with a ratings scale of 1-5. Stability was a common issue for everyone completing this homework. Most of the computations are made in log space and there might be warnings about division near zero. Numpy broadcasting is also used almost everywhere for speed, otherwise the algorithm convergence would take ages (current runtime for five initialisations around 2 seconds with K=1 and two minute with K=12). 

Unit testing was done against exercises in the Edx platform. 

***


### A little bit more detailed description and limitations

(The homework also included programming K-means clustering and a naive EM implementation, but they were excluded from the repo to highlight the mixture model via proper EM algorithm.)

A data matrix was given containing movie ratings made by users where the matrix is extracted from a much larger Netflix database. Any particular user has rated only a small fraction of the movies so the data matrix is only partially filled (sparse). The goal is to predict all the remaining entries in the matrix and thus help recommend movies with better accuracy.

The model assumes that each user's rating profile is a sample from a mixture model, i.e. there are assumed to be K possible types of users and for each user 1) a user type and 2) the rating profile from the Gaussian distribution associated to the type, are sampled. 

The workhorse behind the model is the Expectation Maximisation (EM) algorithm which is used to estimate the mixture from the sparse ratings matrix. The EM algorithm proceeds by iteratively assigning users to a type (E-step) and subsequently estimating the Gaussians associated with each type (M-step). Once the mixture is estimate, it is possible to predict values for all the missing entries in the matrix. 

The problem is made easier by the simplifying but not very restricting assumption that the K types are independent of each other (someone liking action movies has no effect on another viewer on the other side of the planet enjoying romantic comedies), thus it is only required to estimate the proportion of each type and their means and variances. 

Of course, one could argue that the ratings are not normally distributed and the performance of the model could for sure be improved upon finding more representative probability distributions. However, the exercise was already quite heavy with derivations and especially programming the model with numpy broadcasting, thus it is left as is with OKish performance for such a simple model. 







