from known_boundary.GP import optimise,optimise_warp,optimise_warp_no_boundary
from known_boundary.utlis import Trans_function, get_initial_points,transform
from known_boundary.acquisition_function import EI_acquisition_opt,MES_acquisition_opt,Warped_TEI2_acquisition_opt,LCB_acquisition_opt,ERM_acquisition_opt,Warped_EI_acquisition_opt,Warped_TEI1_acquisition_opt
from obj_functions.obj_function import XGBoost
import numpy as np
import matplotlib.pyplot as plt
import GPy
import torch
import botorch
from botorch.utils.transforms import unnormalize,normalize

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double



standard_bounds =np.array([0.,1.]*6).reshape(-1,2) 
n_init = 18
iter_num = 30
N = 20

print('EI')
BO_EI = []

for exp in range(N):
    
    print(exp)

    seed = exp
    
    fun = XGBoost('breast',seed=exp)
    dim = fun.dim
    bounds = fun.bounds
    
    fstar = 100
    fun = Trans_function(fun,fstar,min=False)
    

    X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
    Y_BO = torch.tensor(
        [fun(x) for x in X_BO], dtype=dtype, device=device
    ).reshape(-1,1)


    best_record = [Y_BO.min().item()]
    
    np.random.seed(1234)

    for i in range(iter_num):
        
            #print(i)
        
            train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
            train_X = normalize(X_BO, bounds)
            
            minimal = train_Y.min().item()
            
            train_Y = train_Y.numpy()
            train_X = train_X.numpy()
            
            # train the GP
            res = optimise(train_X,train_Y)
            # print('lengthscale is: ',np.sqrt(res[0])) 
            # print('variance is: ',res[1])
            kernel = GPy.kern.RBF(input_dim=dim,lengthscale= np.sqrt(res[0]),variance=res[1]) 
            m = GPy.models.GPRegression(train_X, train_Y,kernel)
            m.Gaussian_noise.variance.fix(10**(-5))

            standard_next_X = EI_acquisition_opt(m,bounds=standard_bounds,f_best=minimal)
            X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
            Y_next = fun(X_next).reshape(-1,1)

            # Append data
            X_BO = torch.cat((X_BO, X_next), dim=0)
            Y_BO = torch.cat((Y_BO, Y_next), dim=0)
            
            best_record.append(Y_BO.min().item())
            #print(best_record)
            
    best_record = fstar-(np.array(best_record))
    BO_EI.append(best_record)
    
np.savetxt('exp_res/Breast_GP_EI', BO_EI, delimiter=',')




print('MES')
BO_MES = []

for exp in range(N):

    seed = exp
    
    print(exp)
    
    fun = XGBoost('breast',seed=exp)
    dim = fun.dim
    bounds = fun.bounds
    
    fstar = 100
    fun = Trans_function(fun,fstar,min=False)


    X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
    Y_BO = torch.tensor(
        [fun(x) for x in X_BO], dtype=dtype, device=device
    ).reshape(-1,1)
    
    fstar_mes = 0. 

    best_record = [Y_BO.min().item()]
    
    np.random.seed(1234)

    for i in range(iter_num):
        
            train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
            train_X = normalize(X_BO, bounds)
            
            
            fstar_standard = (fstar_mes - Y_BO.mean()) / Y_BO.std()
            fstar_standard = fstar_standard.item()
            
            train_Y = train_Y.numpy()
            train_X = train_X.numpy()
            
            # train the GP
            res = optimise(train_X,train_Y)
            kernel = GPy.kern.RBF(input_dim=dim,lengthscale= np.sqrt(res[0]),variance=res[1]) 
            m = GPy.models.GPRegression(train_X, train_Y,kernel)
            m.Gaussian_noise.variance.fix(10**(-5))

            standard_next_X = MES_acquisition_opt(m,standard_bounds,fstar_standard)
            X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
            Y_next = fun(X_next).reshape(-1,1)

            # Append data
            X_BO = torch.cat((X_BO, X_next), dim=0)
            Y_BO = torch.cat((Y_BO, Y_next), dim=0)
            
            best_record.append(Y_BO.min().item())
            
    best_record = fstar-(np.array(best_record))
    BO_MES.append(best_record)
    
np.savetxt('exp_res/Breast_GP+MES', BO_MES, delimiter=',')


print('ERM')

BO_ERM = []

for exp in range(N):

  print(exp)  
  seed = exp
  
  fun = XGBoost('breast',seed=exp)
  dim = fun.dim
  bounds = fun.bounds
  
  fstar = 100
  fun = Trans_function(fun,fstar,min=False)

  fstar0 = 0.
  Trans = False

  X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
  Y_BO = torch.tensor(
          [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)

  best_record = [Y_BO.min().item()]
  
  np.random.seed(1234)

  for i in range(iter_num):

    #print(i)
    train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
    train_X = normalize(X_BO, bounds)
    
    train_Y = train_Y.numpy()
    train_X = train_X.numpy()
    

    fstar_standard = (fstar0 - Y_BO.mean()) / Y_BO.std()
    fstar_standard = fstar_standard.item()
    
    if not Trans:
      minimal = np.min(train_X)
      res = optimise(train_X,train_Y)
      kernel = GPy.kern.RBF(input_dim=dim,lengthscale= np.sqrt(res[0]),variance=res[1]) 
      m = GPy.models.GPRegression(train_X, train_Y,kernel)
      m.Gaussian_noise.variance.fix(10**(-5))

      standard_next_X = EI_acquisition_opt(m,bounds=standard_bounds,f_best=minimal)
      
      beta = np.sqrt(np.log(train_X.shape[0]))
      _,lcb = LCB_acquisition_opt(m,standard_bounds,beta)
      if lcb < fstar_standard:
        Trans = True
      
    else:
      
      #print('transfromed GP')
              
      train_Y_transform = transform(y=train_Y,fstar=fstar_standard)
      mean_temp = np.mean(train_Y_transform)
      
      res = optimise(train_X,(train_Y_transform-mean_temp))
      kernel = GPy.kern.RBF(input_dim=dim,lengthscale= np.sqrt(res[0]),variance=res[1]) 
      m = GPy.models.GPRegression(train_X, train_Y_transform-mean_temp,kernel)
      m.Gaussian_noise.variance.fix(10**(-5))
      standard_next_X,erm_value = ERM_acquisition_opt(m,bounds=standard_bounds,fstar=fstar_standard,mean_temp=mean_temp)
    
    
    X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)     
    Y_next = fun(X_next).reshape(-1,1)

    # Append data
    X_BO = torch.cat((X_BO, X_next), dim=0)
    Y_BO = torch.cat((Y_BO, Y_next), dim=0)

    best_value = float(Y_BO.min())
    best_record.append(best_value)


  best_record = fstar-(np.array(best_record))
  BO_ERM.append(best_record)
  
np.savetxt('exp_res/Breast_transformedGP+ERM', BO_ERM, delimiter=',')


print('logTEI')
Warped_BO_TEI2 = []

for exp in range(N):

    seed = exp
    
    print(exp)
    
    fun = XGBoost('breast',seed=exp)
    dim = fun.dim
    bounds = fun.bounds
    
    fstar = 100
    fun = Trans_function(fun,fstar,min=False)
    

    X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
    Y_BO = torch.tensor(
        [fun(x) for x in X_BO], dtype=dtype, device=device
    ).reshape(-1,1)



    best_record = [Y_BO.min().item()]
    
    np.random.seed(1234)

    for i in range(iter_num):
        
            train_Y = Y_BO.numpy()
            train_X = normalize(X_BO, bounds)
            train_X = train_X.numpy()
            
            # train the GP
            res = optimise_warp(train_X, train_Y)
            lengthscale = np.sqrt(res[0])
            variance = res[1]
            c = res[2]
            
            
            warp_Y = np.log(train_Y+c)
            mean_warp_Y = np.mean(warp_Y) # use to predict mean
            warp_Y_standard = warp_Y-mean_warp_Y
            
            
            kernel = GPy.kern.RBF(input_dim=dim,lengthscale= lengthscale,variance=variance)  
            m = GPy.models.GPRegression(train_X, warp_Y_standard,kernel)
            m.Gaussian_noise.variance.fix(10**(-5))
            
            standard_next_X = Warped_TEI2_acquisition_opt(model=m,bounds=standard_bounds,f_best=best_record[-1],c=c,f_mean=mean_warp_Y)
            X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
            Y_next = fun(X_next).reshape(-1,1)

            # Append data
            X_BO = torch.cat((X_BO, X_next), dim=0)
            Y_BO = torch.cat((Y_BO, Y_next), dim=0)
            
            best_record.append(Y_BO.min().item())
            #print(best_record[-1])
            
    best_record = fstar-(np.array(best_record))     
    Warped_BO_TEI2.append(best_record)
    
np.savetxt('exp_res/Breast_logGP+logTEI', Warped_BO_TEI2, delimiter=',')


Warped_BO_TEI = []

for exp in range(N):

    seed = exp
    
    print(exp)
    
    fun = XGBoost('breast',seed=exp)
    dim = fun.dim
    bounds = fun.bounds
    
    fstar = 100
    fun = Trans_function(fun,fstar,min=False)
    

    X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
    Y_BO = torch.tensor(
        [fun(x) for x in X_BO], dtype=dtype, device=device
    ).reshape(-1,1)



    best_record = [Y_BO.min().item()]
    
    np.random.seed(1234)
    
    fstar_standard = 0.
    fstar_temp = 0.
    for i in range(iter_num):
        
            train_Y = Y_BO.numpy()
            train_X = normalize(X_BO, bounds)
            train_X = train_X.numpy()
            
            standard_best_record = [np.min(train_Y)]
            # train the GP
            delta2 = 0.3
            delta2_standard =  delta2  #delta2/ Y_BO.std()
            
            res = optimise_warp_no_boundary(train_X, train_Y,-fstar_standard+delta2_standard)

            #res = optimise_warp(train_X, train_Y)
            lengthscale = np.sqrt(res[0])
            variance = res[1]
            c = res[2]
            
            
            warp_Y = np.log(train_Y+c)
            mean_warp_Y = np.mean(warp_Y) # use to predict mean
            warp_Y_standard = warp_Y-mean_warp_Y
            
            
            kernel = GPy.kern.RBF(input_dim=dim,lengthscale= lengthscale,variance=variance)  
            m = GPy.models.GPRegression(train_X, warp_Y_standard,kernel)
            m.Gaussian_noise.variance.fix(10**(-5))
            
            c_unstandard = -c # -c*Y_BO.std()+Y_BO.mean()
            print('C is: ',c_unstandard)
            if c_unstandard>=fstar_temp:
                #print('logEI')
                standard_next_X = Warped_EI_acquisition_opt(model=m,bounds=standard_bounds,f_best=standard_best_record[-1],c=c,f_mean=mean_warp_Y)
            else:
                #print('logTEI')
                standard_next_X = Warped_TEI1_acquisition_opt(model=m,bounds=standard_bounds,f_best=standard_best_record[-1],c=c,f_mean=mean_warp_Y,fstar=fstar_standard)
            
            standard_next_X = Warped_TEI2_acquisition_opt(model=m,bounds=standard_bounds,f_best=best_record[-1],c=c,f_mean=mean_warp_Y)
            X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
            Y_next = fun(X_next).reshape(-1,1)

            # Append data
            X_BO = torch.cat((X_BO, X_next), dim=0)
            Y_BO = torch.cat((Y_BO, Y_next), dim=0)
            
            best_record.append(Y_BO.min().item())
            #print(best_record[-1])
            
    best_record = fstar-(np.array(best_record))     
    Warped_BO_TEI2.append(best_record)
    
np.savetxt('exp_res/Breast_logGP+logTEI(general bounday)', Warped_BO_TEI, delimiter=',')

print('GP-TEI')
BO_TEI = []

for exp in range(N):
    
    print(exp)

    seed = exp
    
    fun = XGBoost('breast',seed=exp)
    dim = fun.dim
    bounds = fun.bounds
    
    fstar = 100
    fun = Trans_function(fun,fstar,min=False)
    

    X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
    Y_BO = torch.tensor(
        [fun(x) for x in X_BO], dtype=dtype, device=device
    ).reshape(-1,1)
    
    fstar0 = 0.


    best_record = [Y_BO.min().item()]
    
    np.random.seed(1234)

    for i in range(iter_num):
        
            train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
            train_X = normalize(X_BO, bounds)
            
            fstar_standard = (fstar0 - Y_BO.mean()) / Y_BO.std()
            fstar_standard = fstar_standard.item()
            
            minimal = train_Y.min().item()
            
            train_Y = train_Y.numpy()
            train_X = train_X.numpy()
            
            # train the GP
            res = optimise(train_X,train_Y)
            # print('lengthscale is: ',np.sqrt(res[0])) 
            # print('variance is: ',res[1])
            kernel = GPy.kern.RBF(input_dim=dim,lengthscale= np.sqrt(res[0]),variance=res[1]) 
            m = GPy.models.GPRegression(train_X, train_Y,kernel)
            m.Gaussian_noise.variance.fix(10**(-5))

            standard_next_X = EI_acquisition_opt(m,bounds=standard_bounds,f_best=minimal,f_star=fstar_standard)
            X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
            Y_next = fun(X_next).reshape(-1,1)

            # Append data
            X_BO = torch.cat((X_BO, X_next), dim=0)
            Y_BO = torch.cat((Y_BO, Y_next), dim=0)
            
            best_record.append(Y_BO.min().item())
            #print(best_record)
            
    best_record = fstar-(np.array(best_record))
    BO_TEI.append(best_record)
    
    
np.savetxt('exp_res/Breast_GP+TEI', BO_TEI, delimiter=',')
        