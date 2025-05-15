import numpy as np
import copy
import pandas as pd
import random
import multiprocessing
from multiprocessing import Process
import time
from .cfdata import CFData


#TODO private data interface
#TODO add option in cfgenerator for changing and reseting feature weights
#TODO docstring
#TODO move all non-hyperparameters away from init to run function
#TODO rename functions. 
class CFGenerator:
    def __init__(self,cf_instance,desired_range,cf_data,outcome_func,elite_count=300,pop_size=3000,
                max_iterations = 150, threshold = 0.05,raw_to_output=True,features_ignored=None,significant_places=1):
        self.outcome_func = outcome_func
        #must be array
        self.desired_range =  desired_range
        if isinstance(cf_instance,int):
            self.original = cf_data[cf_instance]
        else:
            #passed as pd dataframe
            self.original = cf_data.to_raw_format(cf_instance)[0]
        self.data = cf_data
       
        #number of selected top candidates in each run
        self.elite_count= elite_count
        #count of copied original instances
        self.pop_size = pop_size
        #maximum iterations of algorithm before stopping
        self.max_iterations = max_iterations
        #max distance between two generation fitnesses before starting countdown to stop
        self.threshold = threshold
        self.feature_length = cf_data.feature_length
        
        #wether to pass the data in original format or internal data format
        self.raw_to_output = raw_to_output
        
        #number of significant places of floats. prevents candidates that only have epsilon distance
        if isinstance(significant_places,int):
            self.significant_places = significant_places
        else:
            raise SystemError("Not implemented")
        
        self.features_to_vary = np.array(range(self.feature_length))
        
        
        #is set to true if at least one of the number of requested counterfactuals has not actually reached the target intervall
        self.failed = False
        
        #which features to ignore while running alogrithm
        #IDEA multiple runs in which some features are ignored (different in each run) for truly diverse CFs
        ind_to_del = []
        if features_ignored != None:
            for i in features_ignored:
                x = self.data.column_names.index(i)
                ind_to_del.append(x)
        self.features_to_vary = np.delete(self.features_to_vary, ind_to_del)
        np.seterr(divide='ignore')
        self.num_cfs_found=0
        self.cf_found_percentage=0
    
    
    #helper function to allow queue to be passed
    def _process_helper(self,queue,cf_count,l1_weight,l2_weight,features_changed_weight,diversity_loss_weight,cat_fac):
        x = self.single_thread_generate_counterfactuals(cf_count,l1_weight,l2_weight,features_changed_weight,diversity_loss_weight,cat_fac,single_mode=False)
        #shared queue. thread safe with true => ensured that all 
        queue.put(x,True)
    
    
    def multi_thread_generate_counterfactuals(self,cf_count,num_threads=-1,l1_weight=1,l2_weight=0,features_changed_weight=0,
                                              diversity_loss_weight=0,cat_fac=0.5,posthoc_diversity_weight = 1,posthoc_column_weight=0.5,num_posthoc_considered=-1):
        """Generate counterfactuals using multiple threads
        
        Parameters
        ----------
        cf_count : int
            number of desired counterfactuals
        
        Optional Parameters
        ---------
        num_threads : int
            number of threads. is number of cpu cores if not set
        l1_weight : float
            weight of l1 metric
        l2_weight : float
            weight of l2 metric
        features_changed_weight : float
            weight of penalty for changing features. The higher the less features are changed
        diversity_loss_weight : float
            Deprecated. Add penalty for features being equal in different CFs
        cat_fac : float
            Penalty for categorical features being different from the original
        posthoc_diversity_weight : float
            weight of diversity loss in calculation after run. diversifies end result
        posthoc_column_weight : float
            weight of penalty for having changed the same column in different CFs. diversifies end result
        num_posthoc_considered : int
            number of different CFs considered in posthoc diversity
        """
        if num_threads == -1:
            num_threads = multiprocessing.cpu_count()
        procs = []
        queue = multiprocessing.Queue()
        
        #start sub-processes
        for i in range(num_threads):
            p = Process(target=self._process_helper, args=(queue,cf_count,l1_weight,l2_weight,features_changed_weight,diversity_loss_weight,cat_fac))
            procs.append(p)
            p.start()
        #wait for sub-processes to finish
        for i in range(num_threads):
            procs[i].join()
        
        #collect cfs from subprocesses
        cfs = []
        for i in range(queue.qsize()):
            cfs.append(queue.get())
        
        cfs = np.reshape(np.array(cfs),(-1,self.feature_length))
        cfs = np.unique(cfs,axis=0)
        #calculate fitness and order for combined cfs from subprocesses
        fitness = self.calculate_loss(cfs,l1_weight,l2_weight,features_changed_weight,diversity_loss_weight)
        y_loss = self.y_loss(cfs)
        
        
        self.num_cfs_found =0
        self.cf_found_percentage = 0
        no_cfs = np.where(y_loss!=0)[0]
        if len(no_cfs) == len(cfs):
            return "No counterfactuals found!"
        else:
            self.num_cfs_found = len(cfs)-len(no_cfs)
            self.cf_found_percentage = np.round(self.num_cfs_found/len(cfs)*100,2)
            
        
        if num_posthoc_considered==-1:
            num_posthoc_considered=int(1.5*len(cfs))
        fitness += posthoc_diversity_weight*self.diversity_loss(cfs,num_posthoc_considered)
        
        fitness += posthoc_column_weight*self.column_change_loss(cfs,num_posthoc_considered)
        
        indices = np.argsort(fitness)
        cfs = cfs[indices]
        self.raw_cf = cfs[0:cf_count]
        self.raw_pop = cfs
        self.fitness = fitness[indices][:cf_count]
        return self.decode(cfs[:cf_count])
        
        
    
    #main loop
    def single_thread_generate_counterfactuals(self,cf_count,l1_weight=1,l2_weight=0,features_changed_weight=0,diversity_loss_weight=0,cat_fac = 0.3,posthoc_sparsity=False,single_mode=True):
        if len(self.features_to_vary)==0:
            return "No features selected for variation!"
        #useful for debugging. makes graph after run possible
        if single_mode:
            self.plot_y = []
            self.plot_x = []
            self.i=0
        
        #only useful for single run. causes same result for multithread results
        #np.random.seed(int(time.time()))
        #random.seed(int(time.time()))
        
        population = np.full((self.pop_size,len(self.original)),self.original)
        population = population.astype(float)
        self.mutate_and_mate(population,0,0)
        
        iterations = 0
        previous_best_loss = -np.inf
        current_best_loss = np.inf
        stop_cnt = 0
        num_cfs = 0
        while iterations<self.max_iterations:
            #if distance of losses in each generation is <threshold start counting countdown. if distance stays smaller for 10 iterations stop algorithm 
            if abs(previous_best_loss - current_best_loss) <= self.threshold and num_cfs >= cf_count:
                stop_cnt+=1
            else:
                stop_cnt=0
            if stop_cnt >=10:
                break;
            
            previous_best_loss = current_best_loss
            self.mutate_and_mate(population,cf_count,0.6)
            old_population = np.unique(population,axis=0)
            fitness = self.calculate_loss(old_population,l1_weight,l2_weight,features_changed_weight,diversity_loss_weight,cat_fac)
            num_cfs = np.sum(np.greater_equal(50,fitness))
            indices = np.argsort(fitness)
            old_population = old_population[indices]
            fitness = fitness[indices]
            current_best_loss = np.sum(fitness[0:cf_count])
            
            
            if single_mode:
                self.plot_y.append(current_best_loss)
                self.plot_x.append(self.i)
                self.i+=1

            elite_index=0
            temp_elite_count = self.elite_count if len(old_population)>self.elite_count else len(old_population)
            #distribute best candidates among population
            for i in range(0, self.pop_size):
                population[i] = np.copy(old_population[elite_index])
                elite_index += 1
                if elite_index== temp_elite_count:
                    elite_index=0
            iterations+=1

        if posthoc_sparsity:
            loss = self.diversity_loss(old_population[:num_cfs],cf_count)
            indices = np.argsort(loss)
            old_population = old_population[indices]
        
        counterfactuals = old_population[0:cf_count]
        self.raw_cf = old_population[0:cf_count]
        self.raw_pop = old_population
        
        self.num_cfs_found = num_cfs
        self.cf_found_percentage = np.round(num_cfs/cf_count*100,3)
        if single_mode:
            if num_cfs == 0:
                return "No counterfactuals found"
            else:
                return self.decode(counterfactuals)
        else:
            return counterfactuals

    
    #either mutate (change one feature) or mate (exchange one feature with another point)
    def mutate_and_mate(self, datapoints,cf_count,mate_prob):
        change_index = np.random.choice(self.features_to_vary,size=len(datapoints))
        for i in range(cf_count,len(change_index)):
            prob = random.random()
            if prob > mate_prob:
                    upper = self.data.range[change_index[i]][1]
                    lower = self.data.range[change_index[i]][0]
                    if change_index[i] in self.data.cat_indices or change_index[i] in self.data.int_indices:
                        datapoints[i][change_index[i]] = np.random.randint(lower,upper+1)
                    else:
                        datapoints[i][change_index[i]] = np.round(np.random.uniform(lower,upper),self.significant_places)
            else:
                parent2 = np.random.randint(len(datapoints))
                datapoints[i][change_index[i]] = datapoints[parent2][change_index[i]] 
        return datapoints
    
    
    def calculate_loss(self,datapoints,l1_weight,l2_weight,features_changed_weight,diversity_loss_weight,cat_fac=0.3):
        datapoints = np.copy(datapoints)
        
        # l1_weight = 1
        # l2_weight = 0
        # features_changed_weight = 0
        # diversity_loss_weight = 0
        
        
        #when calculating loss categorical data is transformed to 1 (different from original) and 0
        #cat_fac is a factor which can be applied which rewards or punishes changing categoricals.
        #TODO add the option to transform categorical column to int column to allow different level of difficulty 
        #i.e. from bachelor to master is easier than from school to bachelor
        
        #wether 
        loss = self.y_loss(datapoints)
        loss += features_changed_weight*self.features_changed_loss(datapoints,inverse=False) if features_changed_weight > 0 else 0
        
        #set categorical variables to either cat_fac (when different) or 0 for correct calculation of norm
        datapoints= datapoints.transpose()
        i = (np.not_equal(self.original[self.data.cat_indices],datapoints[self.data.cat_indices].transpose()).astype(float)*cat_fac).transpose()
        datapoints[self.data.cat_indices]=i
        datapoints=datapoints.transpose()
        #end
        
        loss += l1_weight*self.distance_loss(datapoints,self.data.feature_weights,1) if l1_weight > 0 else 0
        loss += l2_weight*self.distance_loss(datapoints,self.data.feature_weights,2) if l2_weight > 0 else 0
        loss += diversity_loss_weight*self.diversity_loss(datapoints,len(datapoints)-1) if diversity_loss_weight > 0 else 0
        return loss
    
    def column_change_loss(self,datapoints,no_to_compare=10):
        datapoints = np.equal(self.original,datapoints)
        col_loss = []
        for i in range(1,no_to_compare+1):
            mat = np.equal(datapoints, np.roll(datapoints,i,axis=0))
            col_loss.append(self.feature_length-np.sum(mat,axis=1))
        col_loss = np.sum(col_loss,axis=0)/no_to_compare
        return 1-col_loss/self.feature_length
    
    def diversity_loss(self,datapoints,no_to_compare = 10):
        div_loss = []
        to_shift = copy.deepcopy(datapoints)
        for i in range(1,no_to_compare+1):
            mat = np.equal(datapoints, np.roll(to_shift,i,axis=0))
            div_loss.append(np.sum(mat,axis=1))
        div_loss=np.sum(div_loss,axis=0)/no_to_compare
        
        return div_loss/self.feature_length
    
    #wether datapoint is different or not from original
    def y_loss(self,datapoints):
        data_temp = self.data.backtransform_point(datapoints)
        x = self.outcome_func(data_temp)
        if self.data.target_data.type=="cat":
            x = self.data.target_data._column_transformer.transform(x)
        yloss = 1/(np.logical_and(x >= self.desired_range[0],x <= self.desired_range[1])).astype(float)-1
        yloss[yloss==np.inf]=100
        return yloss
    
    #how many features were changed
    def features_changed_loss(self,datapoints,inverse=False):
        div_loss = None
        if inverse:
            div_loss = np.sum(np.equal(self.original,datapoints),axis=1)
        else:
            div_loss = np.sum(np.not_equal(self.original,datapoints),axis=1)
        return div_loss/len(self.original)
    
    #normalized l^x norm between datapoints and original
    def distance_loss(self,datapoints,weights,order):

        x1hat = (datapoints-self.data.range.T[0])/self.data.span
        x2hat = (self.original-self.data.range.T[0])/self.data.span
        x2hat[self.data.cat_indices]=0

        dist = np.linalg.norm(weights*(x1hat-x2hat),order,axis=1)
        dist /= np.sum(weights)
        return dist
    
    #convert table in internal data format to pandas dataframe with "-" where original and datapoint are same
    def decode(self,cfs):
        """
        Convert table in internal data format to pd dataframe
        
        Parameters
        ----------
        cfs : numpy array
            data to be transformed
        
        Extended summary
        ----------
        Sets "-" where original and data are same. Also adds outcome to all rows. 
        """
        cfs = np.concatenate((np.array([self.original]),cfs),axis=0)

        data_temp = self.data.backtransform_point(cfs)
        outcome = self.outcome_func(data_temp)
        eq = []
        for i in cfs:
            eq.append(np.equal(self.original,i))
        counterfactuals = self.data.backtransform_point(cfs)
        for i in range(len(cfs[0])):
            for j in range(1,len(cfs)):
                if eq[j][i] == True:
                    counterfactuals.iloc[j,i]="-"
        counterfactuals[self.data.outcome] = outcome
        return counterfactuals
    


        
        
    

