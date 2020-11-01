import numpy as np
import common
import em


#%%

netflix_incomplete = np.loadtxt('netflix_incomplete.txt')
titles = ['$K = 1$', '$K = 12$']

#%%
for i, k in enumerate([1, 12]):
    
    title = titles[i]
    mixture = None
    post = None
    cost = None
    
    # try 5 different initialisations
    for j in range(5):
        
        init_mixture, init_post = common.init(netflix_incomplete, K=k, seed=j)
        new_mixture, new_post, new_cost = em.run(netflix_incomplete, init_mixture, init_post)
        
        
        if cost is None or new_cost > cost:
            
            cost = new_cost
            mixture = new_mixture
            post = new_post
            
    print('Cost for', title, 'is:')
    print(cost)
    
    
#%%
X_gold = np.loadtxt('netflix_complete.txt')
X_pred = em.fill_matrix(netflix_incomplete, mixture)
print('\nThe root-mean-square-error (RMSE) with K=12 for the test set is:')
print(common.rmse(X_gold, X_pred))


    


    