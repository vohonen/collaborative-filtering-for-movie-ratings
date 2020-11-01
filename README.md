# Collaborative filtering for Netflix movie ratings
This is a homework from the MIT OCW's massive open online course (MOOC) "Machine Learning with Python". The idea was to build a mixture model almost from scratch to predict Netflix movie ratings from a sparse data matrix using a Gaussian distribution. 

A data matrix was given containing movie ratings made by users where the matrix is extracted from a much larger Netflix database. Any particular user has rated only a small fraction of the movies so the data matrix is only partially filled (sparse). The goal is to predict all the remaining entries in the matrix and thus help recommend movies with better accuracy.

The model assumes that each user's rating profile is a sample from a mixture model, i.e. there are assumed to be $K$ possible types of users and for each user 1) a user type and 2) the rating profile from the Gaussian distribution associated to the type, are sampled. 

The workhorse behind the model is the Expectation Maximisation (EM) algorithm which is used to estimate the mixture from the sparse ratings matrix. The EM algorithm proceeds by iteratively assigning users to a type (E-step) and subsequently estimating the Gaussians associated with each type (M-step). Once the mixture is estimate, it is possible to predict values for all the missing entries in the matrix. 




