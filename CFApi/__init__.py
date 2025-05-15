import inspect
import pandas as pd
import numpy as np
import warnings
import copy
# from .CounterfactualBackend.helpers import FakePredictor
from .CounterfactualBackend.CounterFactualGenerator import CFGenerator as cf
from .CounterfactualBackend.cfdata import DatasetMetadata


#TODO create fakepredictor if prediction is not class but function4
#TODO fix ints not being compactified
#TODO create shadow categoricals
class CFGenerator:
    def __init__(self, dataset, target_variable, pred_func):
        has_pred = getattr(pred_func, "predict", None)
        if callable(has_pred):
            pred_func = pred_func.predict
        self.target_variable = target_variable
        self.metadata = DatasetMetadata(dataset,target_variable)
        self.pred_func = pred_func
        
        self.generator_options = {"elite_count":300,"pop_size":3000,"max_iterations" : 150, "threshold" : 0.05, "significant_places":1}
        
        self.general_run_arguments = {"l1_weight":1,"l2_weight":0, "features_changed_weight":0,"diversity_loss_weight":0,"cat_fac":0.5}
        
        self.multi_thread_arguments = {"num_threads":-1,"posthoc_diversity_weight" : 1,"posthoc_column_weight":0.5,"num_posthoc_considered":-1}
    
    def get_metadata(self):
        return self.metadata
    
    
    def generate_counterfactuals(self, instance, desired_range = None, num_cfs=10, multithreaded=False,**kwargs):
        if isinstance(instance, pd.Series):
            instance = pd.DataFrame(instance).T
        
        if self.target_variable in list(instance):
            instance.drop(self.target_variable,axis=1,inplace=True)
            
        eps = np.finfo(np.float32).eps
        
        self.metadata._calculate_counterfactual_info()
        
        if desired_range == None:
            if self.metadata.target_data.type != "cat":
                raise Exception("Can't infer desired target because it is numerical. Specify the 'desired_range' when calling this function."
                                "If the target isn't numerical change dtype in metadata.")
            else:
                target_values = list(copy.copy(self.metadata.target_data.values))
                if len(target_values)>2:
                    warnings.warn("Opposite class of target variable is ambiguous. Consider specifying 'desired_range'")
                current_outcome = self.pred_func(instance)
                target_values.remove(current_outcome[0])
                desired_range =target_values[0]
        
        if not isinstance(desired_range, list):
            if self.metadata.target_data.type=="cat":
                le = self.metadata.target_data._column_transformer
                transformed_instance = le.transform([desired_range])
                desired_range = [transformed_instance[0]-0.2,transformed_instance[0]+0.2]
            else:
                desired_range = [desired_range-eps,desired_range+eps]
        cf_gen = cf(instance, desired_range, self.metadata, self.pred_func, features_ignored = self.metadata.ignored_features,**self.generator_options)
        if not multithreaded:
            counterfactuals = cf_gen.single_thread_generate_counterfactuals(num_cfs,**self.general_run_arguments)
        else:
            counterfactuals = cf_gen.multi_thread_generate_counterfactuals(num_cfs,**self.general_run_arguments,**self.multi_thread_arguments)
        self.num_cfs_found = cf_gen.num_cfs_found
        self.cf_found_percentage = cf_gen.cf_found_percentage
        return counterfactuals
    
