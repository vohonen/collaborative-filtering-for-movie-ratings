import numpy as np
import kmeans
import common
import naive_em
import em

#%%
X = np.loadtxt("toy_data.txt")
titles = ['$K = 1$', '$K = 2$', '$K = 3$', '$K = 4$']

    
# test K = [1, 2, 3, 4]
for i in range(1,5):
    
    title = titles[i-1]
    mixture = None
    post = None
    cost = 100000
    
    # try 5 different initialisations
    for j in range(5):
        
        init_mixture, init_post = common.init(X, K=i, seed=j)
        new_mixture, new_post, new_cost = kmeans.run(X, init_mixture, init_post)
        
        if new_cost < cost:
            
            cost = new_cost
            mixture = new_mixture
            post = new_post
            
    print('Cost for', title, 'is:')
    print(cost)
    common.plot(X, mixture, post, title)
    
    
#%%

# test K = [1, 2, 3, 4]
for i in range(1,5):
    
    title = titles[i-1]
    mixture = None
    post = None
    cost = None
    
    # try 5 different initialisations
    for j in range(5):
        
        init_mixture, init_post = common.init(X, K=i, seed=j)
        new_mixture, new_post, new_cost = naive_em.run(X, init_mixture, init_post)
        
        if cost is None or new_cost > cost:
            
            cost = new_cost
            mixture = new_mixture
            post = new_post
            
    print('Cost for', title, 'is:')
    print(cost)
    common.plot(X, mixture, post, title)
    
#%%

for i in range(1,5):
    
    title = titles[i-1]
    mixture = None
    post = None
    bic = None
    
    # try 5 different initialisations
    for j in range(5):
        
        init_mixture, init_post = common.init(X, K=i, seed=j)
        new_mixture, new_post, new_cost = naive_em.run(X, init_mixture, init_post)
        
        new_bic = common.bic(X, new_mixture, new_cost)
        
        if bic is None or new_bic > bic:
            
            bic = new_bic
            mixture = new_mixture
            post = new_post
            
    print('Bic for', title, 'is:')
    print(bic)
    common.plot(X, mixture, post, title)
    
#%%

netflix_incomplete = np.loadtxt('netflix_incomplete.txt')
titles = ['$K = 1$', '$K = 12$']

#%%
for k, i in enumerate([1, 12]):
    
    title = titles[k]
    mixture = None
    post = None
    cost = None
    
    # try 5 different initialisations
    for j in range(5):
        
        init_mixture, init_post = common.init(netflix_incomplete, K=i, seed=j)
        new_mixture, new_post, new_cost = em.run(netflix_incomplete, init_mixture, init_post)
        
        
        if cost is None or new_cost > cost:
            
            cost = new_cost
            mixture = new_mixture
            post = new_post
            
    print('Cost for', title, 'is:')
    print(cost)
    # common.plot(netflix_incomplete, mixture, post, title)
    
    
#%%
X_gold = np.loadtxt('netflix_complete.txt')
X_pred = em.fill_matrix(netflix_incomplete, mixture)
print(common.rmse(X_gold, X_pred))


    


    