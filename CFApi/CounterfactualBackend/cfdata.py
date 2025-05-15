import scipy
from sklearn import preprocessing
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .helpers import deprecated
import warnings
from copy import copy

@dataclass
class VarMetadata:
    name : any
    type : str
    values : list
    is_binary : bool
    _column_transformer : any
    _internal_range : list
    span : list
    feature_weight : float
    _index : float
    is_outcome : bool = False
    is_excluded :bool = False
    
    _init_flag : bool = False

    def __setattr__(self, name, value):
        if name == 'type' and self._init_flag:
            if value == "int" or value == "num":
                raise Exception("Changing variable type to numerical or integer from categorical isn't supported. "
                                "Consider transforming the columns themselves.")
            else:
                raise Exception("That doesn't work. Use modify_values function")
                warnings.warn(f'Variable type is now assumed to be binary. '
                              'If that\'s unintended use function switch_to_categorical')
        if name=="values" and self._init_flag and self.type=="cat":
            raise Exception("Adding or removing possible values for categorical data can only be done by using"
                            "modify_values function.")
        if name=="span" and self._init_flag:
            warnings.warn("Modifying span has no effect")
        # if name.startswith("_") and self._init_flag:
        #     warnings.warn("Changing internal variables is generally a bad idea...")
        if name=="is_outcome" and self._init_flag:
            raise Exception("Outcome variable can only be defined when calculating metadata")
        
        super().__setattr__(name, value)
        
    def switch_to_categorical(self,data):
        raise NotImplementedError("Not finished yet. Transform dataset itself and recalculate metadata.")
        
    def modify_values(self,new_values):
        raise NotImplementedError("Not finished yet. Transform dataset itself and recalculate metadata.")
        
    
        

        
class DatasetMetadata:
    
    def to_raw_format(self, x):
        return self.to_internal_format(x)
    
    def backtransform_point(self,x):
        return self.from_internal_format(x)
    
    def to_internal_format(self,x):
        """
        Transform data point to internal data format. Currently only pandas dataframe is supported.
        """
        x = x.copy()
        for i in self.cat_indices:
            x.iloc[:,i] = self.categorical_transformers[i].transform(x.iloc[:,i])
        return x.to_numpy().astype(float)
    
    def from_internal_format(self,data):
        """
        Transform a point (or points) from internal data format to original format. Currently only
        retransform to pandas dataframe is supported.
        """
        if len(data[0]) == self.feature_length:
            transformed_data = pd.DataFrame(data=data,columns= self.column_names[:-1])
            for i in self.cat_indices:
                integer_vals = transformed_data.iloc[:,i].astype(int)
                transformed_data.iloc[:,i] = self.categorical_transformers[i].inverse_transform(integer_vals)
            return transformed_data
        else:
            raise NotImplmentedError 
    
    def _calculate_counterfactual_info(self):
        self.range = []
        self.feature_weights = []
        self.span = []
        self.categorical_transformers = {}
        self.ignored_features = []
        for i in self.var_metadata.values():
            if i.is_outcome:
                continue
            self.range.append(i._internal_range)
            self.feature_weights.append(i.feature_weight)
            self.span.append(i.span)
            self.categorical_transformers[i._index] = i._column_transformer
            if i.is_excluded:
                self.ignored_features.append(i.name)
        self.range = np.array(self.range)
        self.feature_weights = np.array(self.feature_weights)
        self.span = np.array(self.span)
            
    
    def __init__(self, data, target_variable, dataset_dtype=None):
        if dataset_dtype == None:
            if not isinstance(data,pd.DataFrame):
                raise Exception("The provided data is not a pandas dataframe. Please specify 'dataset_dtype'")
        else:
            raise NotImplmentedError("You can temporarily circumvent by transforming data to pandas dframe")
        
        self.outcome = target_variable
        
        
        
        column_names = list(data)
        self.feature_length = len(column_names)-1
        self.column_names = column_names
        categorical_columns = list(data.select_dtypes(exclude=[np.number]))
        numerical_columns =  data.select_dtypes(include=[float])
        integer_columns = data.select_dtypes(include=[int])
        
        self.cat_indices = [data.columns.get_loc(i) for i in categorical_columns]
        self._reverse_cat = {i:self.column_names[i] for i in self.cat_indices}
        self.int_indices = [data.columns.get_loc(i) for i in integer_columns]
        self.num_indices = [data.columns.get_loc(i) for i in numerical_columns]
        
        self.var_metadata = {}
        for i in column_names:
            unique_values = data[i].unique()
            var_type = "cat" if i in categorical_columns else ("num" if i in numerical_columns else "int")
            is_binary = False
            column_transformer = None
            if var_type == "cat":
                values = unique_values
                if len(unique_values) == 2:
                    is_binary = True
                column_transformer = preprocessing.LabelEncoder()
                column_transformer.fit(data[i])
                internal_range = column_transformer.transform(column_transformer.classes_)
                internal_range = [min(internal_range),max(internal_range)]
                feature_weight = 1
                
            else:
                
                if len(unique_values) == 2:
                    warnings.warn(f"Feature '{i}' has a numerical datatype but seems to be binary. "
                                  "Consider changing it to categorical.")
                    is_binary = True
                elif len(unique_values)<10 and var_type == "num":
                    warnings.warn(f"Feature '{i}' is has float like datatype but doesn't seem to be continouus."
                                  "Check if the feature really is continouus otherwise the algorithm might give undesired results.")
                values = [data[i].min(axis=0),data[i].max(axis=0)]
                internal_range = values
                span = internal_range[1]-internal_range[0]
                normed_data = (data[i]-values[0])/span
                mad = np.median(np.absolute(normed_data - np.median(normed_data)))#scipy.stats.median_absolute_deviation(normed_data,axis=0,scale=1)
                feature_weight = 1
                if mad!=0:
                    feature_weight = 1/mad
                
            span = internal_range[1]-internal_range[0]
            self.var_metadata[i] = VarMetadata(i, var_type, values, is_binary, column_transformer, internal_range, span, feature_weight, column_names.index(i))
            self.var_metadata[i]._init_flag = True
            if i == target_variable:
                self.var_metadata[i].__dict__["is_outcome"]=True
                self.target_data = self.var_metadata[i]
                # self.var_metadata[i].is_outcome = True
                
        self._var_metadata_original = copy(self.var_metadata)
        if self.target_data.type=="cat":
            self.cat_indices.remove(self.target_data._index)
            
            
    def __getitem__(self,key):
        return self.var_metadata[key]
    
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        try:
            from tabulate import tabulate
            arr = []
            for i in self.var_metadata.values():
                if i.type!="cat":
                    vals = f"{np.round(i.values[0],5)} - {np.round(i.values[1],5)}"
                else:
                    strings = [str(l) for l in i.values]
                    vals = ", ".join(strings)
                    if len(vals) > 45:
                        vals = vals[:45]+"..."
                arr.append([i.name,i.type,i.is_binary, vals, i.feature_weight ])
            
            headers = ["name", "type", "is binary","range or values","weight"]
            return tabulate(arr,headers=headers)
        except Exception as e:
            warnings.warn(e)
            ret = ""
            for i in self.var_metadata:
                ret+=str(self.var_metadata[i])+"\n"
            return ret

#Everything from here on just exists for legacy reasons
#----------------------------------
#holds metadata about data like ranges, MAD, which columns are int etc...
class CFData:
    def __init__(self,data,outcome):
        """Generate counterfactuals using multiple threads
        
        Parameters
        ----------
        data : pandas dataframe
            the looked at data
        outcome : string
            name of column which is to be changed outcome
        """
        
        
        #prevent changes in original dataset
        data = data.copy(deep=True)
        
        #minus one because target gets removed
        self.feature_length = len(data.T)-1
        self.outcome = outcome
        self.column_names = list(data.columns)
        
        #move target at end
        self.column_names.remove(outcome)
        self.column_names.append(outcome)
        data = data.loc[:,self.column_names]
        
        self.outcome_is_categorical = False
        
        #which columns are categorical data
        self.categorical_columns = list(data.select_dtypes(exclude=[np.number]))
        self.cat_indices = [data.columns.get_loc(i) for i in self.categorical_columns]
        
        if self.outcome in self.categorical_columns:
            self.outcome_is_categorical=True
        
        column_values = []
        for x in self.column_names:
            if x in self.categorical_columns:
                column_values.append(data[x].unique())
            else:
                column_values.append([])
        #encode categorical data
        self.categorical_encoders = []
        for i in self.categorical_columns:
            le = preprocessing.LabelEncoder()
            data[i] = le.fit_transform(data[i])
            self.categorical_encoders.append(le)
        
        #which columns are floats                            
        numerical_columns =  data.select_dtypes(include=[float])
        self.num_indices = [data.columns.get_loc(i) for i in numerical_columns]
        #which columns are ints
        integer_columns = data.select_dtypes(include=[int])
        self.int_indices = [data.columns.get_loc(i) for i in integer_columns]
        
        
        self.target_range = [data[outcome].min(), data[outcome].max()]
        self.target_span = self.target_range[1]-self.target_range[0]
        
        
        
        self.target_is_cat = not pd.api.types.is_numeric_dtype(data[outcome])
        self.target_is_int = pd.api.types.is_integer_dtype(data[outcome])
        
        if self.target_is_cat:
            self.target_type = "cat"
        elif self.target_is_int:
            self.target_type = "int"
        else:
            self.target_type = "num"
        
        
        if self.target_is_int or self.target_is_cat:
            self.target_values = data[outcome].unique()
        else:
            self.target_values = None
        
        
        
        self.training_data = data
        #remove outcome from data since it is no longer necessary
        self.data = data.drop([outcome],axis=1).to_numpy()
        
        self.range = np.array([self.data.min(axis=0),self.data.max(axis=0)]).transpose()
        
        
        self.span = self.data.max(axis=0)-self.data.min(axis=0)
        
        self.column_description = []
        for i,x in enumerate(self.column_names[:-1]):
            temp = [x]
            if x in self.categorical_columns:
                temp.append("categorical")
                temp.append(column_values[i])
            else:
                temp.append("numerical")
                temp.append(self.range[i])
            self.column_description.append(temp)
            
        self.no_weight = np.ones(len(self.column_names)-1)
        
        #calculate feature weights
        norm_data = (self.data-self.range.T[0])/self.span
        self.feature_weights = scipy.stats.median_absolute_deviation(norm_data,axis=0,scale=1)
        
        #prevent infinity feature weight
        self.feature_weights[self.feature_weights==0]=1
        self.feature_weights[self.cat_indices] = 1
        #feature weight is inverse of mad. high weight=hard to change feature
        self.feature_weights=np.divide(1,self.feature_weights)
        self.original_feature_weights = self.feature_weights.copy()
        



        
    @deprecated()
    def get_feature_range(self):
        return self.range
    
    #for pickling. 
    def __getstate__(self):
        """
        For pickling
        
        Extended summary
        ----------
        Delete big unneccessary data while pickling. 
        Model is presumably already trained when loading from disk.
        """
        state = self.__dict__.copy()
        del state["data"]
        if "training_data" in state:
            del state["training_data"]
        return state

    def __setstate__(self, state):
        """
        Counterpart to __getstate__
        """
        self.__dict__.update(state)
        self.data = np.array(0)

    def __getitem__(self,key):
        return self.data[key]
    
    @deprecated()
    def get_training_data(self):
        """
        Get data in raw format with target. Shortcut for training model
        """
        return self.training_data
    
    def backtransform_point(self,data):
        """
        Transform a point (or points) from internal data format to panda dataframe. Categoricals will be re-transformed to their names.
        """
        if len(data.shape) == 1:
            data = [data]
        transformed_data = pd.DataFrame(data=data,columns= self.column_names[:-1])
        index_max = -1 if self.outcome_is_categorical else len(self.categorical_columns)
        j=0
        for i in self.categorical_columns[:index_max]:
            le = self.categorical_encoders[j]
            transformed_data[i]=transformed_data[i].astype(int)
            transformed_data[i]= le.inverse_transform(transformed_data[i].values)
            j+=1
        return transformed_data
    
    def set_feature_weight(self,name,weight):
        """
        Set feature weight of single column
        
        Parameters
        ----------
        name : string
            name of column
        weight : float
            new weight
        """
        idx = self.column_names.index(name)
        self.feature_weights[idx]=weight
    
    def reset_feature_weights(self):
        """
        Reset all feature weight to inverse MAD
        """
        self.feature_weights = self.original_feature_weights.copy()
        
    def set_feature_weights(self,dic):
        for i in dic:
            self.set_feature_weight(i,dic[i])

        
    def get_feature_weights(self):
        """
        Get the feature weights which are used in metric calculation.
        """
        dic = {}
        for i in range(len(self.column_names)-1):
            dic[self.column_names[i]]=self.feature_weights[i]
        return dic
    
    def to_raw_format(self,x):
        """
        Transform panda data point to internal data format
        """
        x = x.copy()
        j=0
        index_max = -1 if self.outcome_is_categorical else len(self.categorical_columns)

        for i in self.categorical_columns[:index_max]:
            le = self.categorical_encoders[j]
            x[i] = le.transform(x[i])
            j+=1
        return x.to_numpy().astype(float)

    
    def get_feature_type(self,feature_name):
        i = self.column_names.index(feature_name)
        type_str = "num" if i in self.num_indices else "cat"
        if type_str == "num":
            return "num"
        else:
            if i not in self.cat_indices:
                return "int"
            else:
                return "cat"
    def __repr__(self):
        return self.__str__()

    
    def __str__(self):
        arr = []
        for i,x in enumerate(self.column_names[:-1]):
            type_str = "num" if i in self.num_indices else "cat"
            if type_str == "num":
                values = str(self.range[i][0])+"-"+str(self.range[i][1])
            else:
                if i not in self.cat_indices:
                    values = str(self.range[i][0])+"-"+str(self.range[i][1])
                    type_str = "int"
                else:
                    cat_index = self.cat_indices.index(i)
                    values = str(self.categorical_encoders[cat_index].classes_)
            arr2 = [x,type_str,values]
            arr.append(arr2)
            
        if self.target_is_cat:
            arr.append([self.outcome, self.target_type, self.target_values])
        else:
            arr.append([self.outcome, self.target_type, f"{self.target_range[0]} - {self.target_range[1]}"])
        from tabulate import tabulate
        headers = ["variable name", "type", "range or values"]
        return tabulate(arr,headers=headers)
