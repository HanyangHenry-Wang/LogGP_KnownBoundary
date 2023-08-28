from known_boundary.GP import optimise,optimise_warp,optimise_warp_no_boundary
from known_boundary.utlis import Trans_function, get_initial_points,transform
from known_boundary.acquisition_function import EI_acquisition_opt,MES_acquisition_opt,Warped_EI_acquisition_opt,Warped_TEI2_acquisition_opt,LCB_acquisition_opt,ERM_acquisition_opt
import numpy as np
import GPy
import torch
from botorch.test_functions import Ackley,Levy,Beale,Branin,Hartmann,Rosenbrock,Powell
from botorch.utils.transforms import unnormalize,normalize

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

function_information = []


temp={}
temp['name']='Branin2D' 
temp['function'] = Branin(negate=False)
temp['fstar'] =  0.397887
temp['min']=True 
function_information.append(temp)

temp={}
temp['name']='Beale2D' 
temp['function'] = Beale(negate=False)
temp['fstar'] =  0. 
temp['min']=True 
function_information.append(temp)

temp={}
temp['name']='Hartmann3D' 
temp['function'] = Hartmann(dim=3,negate=False)
temp['fstar'] =  -3.86278 
temp['min']=True 
function_information.append(temp)


temp={}
temp['name']='Powell4D' 
temp['function'] = Powell(dim=4,negate=False)
temp['fstar'] = 0.
temp['min']=True 
function_information.append(temp)

temp={}
temp['name']='Rosenbrock5D' 
temp['function'] = Rosenbrock(dim=5,negate=False)
temp['fstar'] = 0.
temp['min']=True 
function_information.append(temp)

temp={}
temp['name']='Ackley7D' 
temp['function'] = Ackley(dim=7,negate=False)
temp['fstar'] = 0.
temp['min']=True 
function_information.append(temp)




for information in function_information:

    fun = information['function']
    dim = fun.dim
    bounds = fun.bounds
    standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
    
    n_init = 4*dim
    iter_num = min(10*dim,60) 
    N = 20

    fstar = information['fstar']
    fun = Trans_function(fun,fstar,min=True)
    
    
    ################################################### GP+EI ###########################################
    BO_EI = []

    for exp in range(N):
        
        print(exp)
        
        seed = exp

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)

        best_record = [Y_BO.min().item()]
        np.random.seed(1234)

        for i in range(iter_num):
            
                train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
                train_X = normalize(X_BO, bounds)
                
                minimal = train_Y.min().item()
                
                train_Y = train_Y.numpy()
                train_X = train_X.numpy()
                
                # train the GP
                res = optimise(train_X,train_Y)
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
                
        best_record = np.array(best_record)+fstar 
        BO_EI.append(best_record)
        
    np.savetxt('exp_res/'+information['name']+'_GP+EI', BO_EI, delimiter=',')
        
    ##################################################### GP+MES ##################################################
    BO_MES = []

    for exp in range(N):

        seed = exp
        
        print(exp)
    
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
                
        best_record = np.array(best_record)+fstar 
        BO_MES.append(best_record)
        
    np.savetxt('exp_res/'+information['name']+'_GP+MES', BO_MES, delimiter=',')
    

################################## logGP+EI (no boundary) ######################################
    
    logBO_no_boundary_logEI = []

    for exp in range(N):
        
        print(exp)

        seed = exp

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
                fstar_standard = (fstar - Y_BO.mean()) / Y_BO.std()
                fstar_standard = fstar_standard.item()
                
                train_Y = train_Y.numpy()
                train_X = train_X.numpy()
                
                standard_best_record = [np.min(train_Y)]
                
                
                # train the GP
                res = optimise_warp_no_boundary(train_X, train_Y,-fstar_standard+1.5)
                lengthscale = np.sqrt(res[0])
                variance = res[1]
                c = res[2]
                
                # print('lengthscale is: ', lengthscale)
                # print('variance is: ',variance)
                # print('c is: ',c)
                # print('check:',-np.min(train_Y)+10**(-5),-np.min(train_Y)-fstar_standard+1.5)
                
                warp_Y = np.log(train_Y+c)
                mean_warp_Y = np.mean(warp_Y) # use to predict mean
                warp_Y_standard = warp_Y-mean_warp_Y
                
                
                kernel = GPy.kern.RBF(input_dim=dim,lengthscale= lengthscale,variance=variance)  #np.sqrt(res[0])
                m = GPy.models.GPRegression(train_X, warp_Y_standard,kernel)
                m.Gaussian_noise.variance.fix(10**(-5))
                
                standard_next_X = Warped_EI_acquisition_opt(model=m,bounds=standard_bounds,f_best=standard_best_record[-1],c=c,f_mean=mean_warp_Y)
                X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
                Y_next = fun(X_next).reshape(-1,1)

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                best_record.append(Y_BO.min().item())
                print(best_record[-1])
                
        best_record = np.array(best_record) +fstar          
        logBO_no_boundary_logEI.append(best_record)
        
        
    np.savetxt('exp_res/'+information['name']+'_logGP(NoBoundary)+logEI', logBO_no_boundary_logEI, delimiter=',')
    
    
    ##################################################### log GP+logEI ##################################################
    
    Warped_BO_logEI = []

    for exp in range(N):

        seed = exp
        
        print(exp)

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
                
                
                standard_next_X = Warped_EI_acquisition_opt(model=m,bounds=standard_bounds,f_best=best_record[-1],c=c,f_mean=mean_warp_Y)
                X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
                Y_next = fun(X_next).reshape(-1,1)

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                best_record.append(Y_BO.min().item())
                #print(best_record[-1])
                
        best_record = np.array(best_record)+fstar         
        Warped_BO_logEI.append(best_record)
        
    np.savetxt('exp_res/'+information['name']+'_logGP+logEI', Warped_BO_logEI, delimiter=',')
    
    
    
    ##################################################### log GP+logTEI ##################################################
    Warped_BO_logTEI = []

    for exp in range(N):

        seed = exp
        
        print(exp)

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
                
        best_record = np.array(best_record)+fstar         
        Warped_BO_logTEI.append(best_record)
        
    np.savetxt('exp_res/'+information['name']+'_logGP+logTEI'+'_0.2', Warped_BO_logTEI, delimiter=',')
    
    
    
    ############################################### ERM ###############################################################
    BO_ERM = []
    for exp in range(N):

        print(exp)  
        seed = exp
        
        fstar0 = 0.
        Trans = False

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
                    [fun(x) for x in X_BO], dtype=dtype, device=device
                ).reshape(-1,1)

        best_record = [Y_BO.min().item()]

        np.random.seed(1234)

        for i in range(iter_num):

            #print(iter)
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


        best_record = np.array(best_record)+fstar
        BO_ERM.append(best_record)
        
    np.savetxt('exp_res/'+information['name']+'_transformedGP+ERM', BO_ERM, delimiter=',')
    
    
    
############################ GP TEI #################################################
    print('GP-TEI')
    BO_TEI = []

    for exp in range(N):
        
        print(exp)

        seed = exp

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
                
        best_record = np.array(best_record)+fstar 
        BO_TEI.append(best_record)
        
    np.savetxt('exp_res/'+information['name']+'_GP+TEI', BO_TEI, delimiter=',')
            