import numpy as np
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b


def cov_RBF(x1, x2, parameters):  
    
    lengthscale_square =   parameters[:,0]  # lengthscale^2
    variance = parameters[:,1] # sigma^2


    if x1.shape[1]!=x2.shape[1]:
        x1=np.reshape(x1,(-1,x2.shape[1]))

    Euc_dist=euclidean_distances(x1,x2)

    return variance*np.exp(-0.5*np.square(Euc_dist)/lengthscale_square)


#################### Normal GP ############################

def log_llk(X,y,parameters):

    noise_delta = 10**(-6)
    
    parameters = parameters.reshape(-1,2)
    
    if np.isnan(parameters).any():
        #print('issue with scipy.minimize!')
        
        return -np.inf

    KK_x_x=cov_RBF(X,X,parameters)+np.eye(len(X))*noise_delta     
    if np.isnan(KK_x_x).any(): #NaN
        print("nan in KK_x_x !")   
        #print('X is: ',X)
        print('parameter is: ',parameters)
        print(np.isnan(parameters).any())

    # try:
    #     L=scipy.linalg.cholesky(KK_x_x,lower=True)
    #     alpha=np.linalg.solve(KK_x_x,y)

    # except: # singular
    #     return -np.inf
    
    try:
        first_term = -0.5*np.log(np.linalg.det(KK_x_x))
        
        KK_inv = np.linalg.inv(KK_x_x)
        second_term = -0.5* np.dot(np.dot(y.T,KK_inv),y)
            

    except: # singular
        return -np.inf

    logmarginal = first_term+second_term -0.5*len(y)*np.log(2*3.1415926)
    
    return logmarginal.item()


def optimise(X, y):

    opts ={'maxiter':1000,'maxfun':200,'disp': False}

    bounds = np.array([[0.015**2,0.6**2],[0.01,10]])
    hyper_num = 2  #lengthscale, variance
    restart_num =  9*hyper_num  
    
    value_holder = []
    candidate_holder = []
    
    for _ in range(restart_num):
      init_hyper = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(60*(hyper_num-1), hyper_num))  
      logllk_holder = [0]*init_hyper.shape[0]
      for ii,val in enumerate(init_hyper):           
          logllk_holder[ii] = log_llk(X,y,val) 
          
      x0=init_hyper[np.argmax(logllk_holder)] # we pick one best value from 50 random one as our initial value of the optimization

      # Then we minimze negative likelihood
      res = minimize(lambda x: -log_llk(X,y,parameters=x),x0,
                                  bounds=bounds,method="L-BFGS-B",options=opts) 

      candidate_holder.append(res.x)
      value_holder.append(log_llk(X,y,res.x))


    best_parameter = candidate_holder[np.argmax(value_holder)]
  
        
    return best_parameter




#################### Log GP ############################
def log_llk_warp(X,y,parameters):

    noise_delta = 10**(-6)
    
    parameters = parameters.reshape(-1,3)
    kernel_parameters = parameters[:,:2]
    c = parameters[:,-1]
    
    if np.isnan(parameters).any():
        
        #print('issue with scipy.minimize!')
        
        return -np.inf
    
    y_temp = np.log(y+c)
    y_temp_mean = np.mean(y_temp)
    y_warp = y_temp-y_temp_mean
    
    
    KK_x_x=cov_RBF(X,X,kernel_parameters)+np.eye(len(X))*noise_delta     
    if np.isnan(KK_x_x).any(): #NaN
        print("nan in KK_x_x !")   

    # try: #check whether it is singular
    #     L=scipy.linalg.cholesky(KK_x_x,lower=True)
    #     alpha=np.linalg.solve(KK_x_x,y_warp)

    # except: # singular
    #     return -np.inf
    
    try:
        first_term = -0.5*np.log(np.linalg.det(KK_x_x))
        
        KK_inv = np.linalg.inv(KK_x_x)
        second_term = -0.5* np.dot(np.dot(y_warp.T,KK_inv),y_warp)
            

    except: # singular
        return -np.inf
    
    
    third_term = (len(X)-1)/len(X) * np.sum( np.log(1/(y+c)) ) 

    logmarginal = first_term+second_term - 0.5*len(y)*np.log(2*3.1415926)  +third_term
    
    return logmarginal.item()


def optimise_warp(X, y,variance_lb = 0.01):

    opts ={'maxiter':1000,'maxfun':200,'disp': False}
    
    bounds = np.array([[0.015**2,0.6**2],[variance_lb,10.],[10**(-5),0.3]]) 
    hyper_num = 3
    restart_num = 9*hyper_num 
    
    value_holder = []
    candidate_holder = []
    
    for _ in range(restart_num):
      init_hyper = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(60*(hyper_num-1), hyper_num))
      logllk_holder = [0]*init_hyper.shape[0]
      for ii,val in enumerate(init_hyper):           
          logllk_holder[ii] = log_llk_warp(X,y,val) 
          
      x0=init_hyper[np.argmax(logllk_holder)] 

      # Then we minimze negative likelihood
      res = minimize(lambda x: -log_llk_warp(X,y,parameters=x),x0,
                                  bounds=bounds,method="L-BFGS-B",options=opts) 

      candidate_holder.append(res.x)
      value_holder.append(log_llk_warp(X,y,res.x))


    best_parameter = candidate_holder[np.argmax(value_holder)]
  
        
    return best_parameter




def optimise_warp_no_boundary(X, y,upper=1,lower=0):

    opts ={'maxiter':1000,'maxfun':200,'disp': False}
    
    ymin = np.min(y)
    
    if lower == 0:
    
        if upper>-ymin+10**(-6):
        
            bounds = np.array([[0.015**2,0.6**2],[0.01,10.],[-ymin+10**(-6),upper]])  
        
        else:
            
            bounds = np.array([[0.015**2,0.6**2],[0.01,10.],[-ymin+10**(-10),upper]])  
            
    else:
        if lower > -ymin+10**(-6):
             bounds = np.array([[0.015**2,0.6**2],[0.01,10.],[lower,upper]])  
        else:
            bounds = np.array([[0.015**2,0.6**2],[0.01,10.],[-ymin+10**(-6),upper]])  
    
    hyper_num = 3
    restart_num = 9*hyper_num 
    
    value_holder = []
    candidate_holder = []
    
    for _ in range(restart_num):
      init_hyper = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(60*(hyper_num-1), hyper_num))
      logllk_holder = [0]*init_hyper.shape[0]
      for ii,val in enumerate(init_hyper):           
          logllk_holder[ii] = log_llk_warp(X,y,val) 
          
      x0=init_hyper[np.argmax(logllk_holder)] # we pick one best value from 50 random one as our initial value of the optimization

      # Then we minimze negative likelihood
      res = minimize(lambda x: -log_llk_warp(X,y,parameters=x),x0,
                                  bounds=bounds,method="L-BFGS-B",options=opts) 

      candidate_holder.append(res.x)
      value_holder.append(log_llk_warp(X,y,res.x))


    best_parameter = candidate_holder[np.argmax(value_holder)]
  
        
    return best_parameter