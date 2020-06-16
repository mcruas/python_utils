#############################################################################
###################### ANALISE EXPLORATORIA #################################
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import pandas as pd
import numpy as np
'''
This function produces plots for all combinations of of a given feature 
'''

def Plots_agregados(
      df, # DataFrame
      x, # Name of variable on x-axis
      var,  # variable that forms y-axis
      var_level1 = None, # Variable to separate plots
      param_to_show = 8, # either a maximum int number or a proportion (0,1), depending on the plot_type value
      min_total = 1, # number of minimum size to show a plot
      figsize = (12,8),
      cut_by = 'number',   #  Choose to cut variables by'number' or 'proportion'
      grouper_x = None, # Use when changing frequency of data
      plot_type = 'count', # Choose between ['count', 'proportion'][1],
      plot_title = "Fonte: {}", # Plot title. Must insert {} on the place of level1
      file_name = 'Data_Concorrente_Fonte_{}',
      fig_extension = 'png',
      show_total_on_title = True,
      ## Iterates over categories of a data frame and plots occurences over time
      base = '',
      path = '',
      format_file_name = False,
      subset_level1 = None # In case only wants a subset of values of level1
):
      if (grouper_x is None) : grouper_x = x
      vector_level1 = df[var_level1].unique() if subset_level1 is None else subset_level1
      for level1 in vector_level1: # level1 = vector_level1[0]
            index_level1 = (df[var_level1] == level1)
            n_level = sum(index_level1)
            if n_level < min_total:
                  print(f'{level1}: Less than {min_total} values')
                  continue;
            counts = df.loc[index_level1, var].value_counts(normalize = True)
            # Defines the shown_list
            if cut_by == 'number':
                  if not isinstance(param_to_show, int):
                        print("[ERROR] Invalid value for param_to_show!")
                        return -1;
                  shown_list = counts[0:param_to_show].index
            elif (param_to_show >= 0) & (param_to_show <= 1):
                  shown_list = counts[counts >= param_to_show].index
                  if len(shown_list) == 0:
                        print(f'{level1}: 0 values on list')
                        continue;
            else:
                  print("[ERROR] Invalid value for param_to_show!")
                  return -1;
            # df.loc[index_level1, 'Concorrente'].value_counts()[0:10]
            arg_value_counts = {} if (plot_type == 'count') else {'normalize':True}
            df_agg = (df[index_level1]
                  .groupby([grouper_x, var_level1])[var]
                  .value_counts(*arg_value_counts)
                  .reset_index(name = plot_type)
                  .sort_values(x)
            )
            df_plot = df_agg[df_agg[var].isin(shown_list)]
            plt.figure(figsize=figsize)
            fig = sns.lineplot(data = df_plot, x=x, y=plot_type, hue=var) \
                        .set_title(plot_title.format(level1) + f' | Total values: {n_level}')
            folder_path = os.path.join(base, path)
            if not os.path.isdir(folder_path):
                  os.mkdir(folder_path)
            file_name_formated = file_name.format(level1)
            if format_file_name:
                  file_name_formated = re.sub(r'\W+', '', file_name_formated)
            fig.get_figure().savefig(os.path.join(folder_path, file_name_formated + '.' + fig_extension))




# Imprime a tabela de estatísticas
def Print_classification_report(model, X_test, y_test, prob_threshold = 0.5):
  y_pred = (model.predict_proba(X_test)[:,1] >= prob_threshold) + 0.
  print(metrics.classification_report(y_test, y_pred))
#   metrics.classification(y_test, y_pred);

  
# Plotar gráfico com a fronteira eficiente de tradeoffs precision-recall
def Classification_report_tradeoff(model, X_test, y_test, range_threshold = [0.2,0.9]):  
  range_threshold_lin = np.linspace(range_threshold[0], range_threshold[1])
  vector_precision = np.zeros(len(range_threshold_lin))
  vector_recall = np.zeros(len(range_threshold_lin))
  for i, thr in enumerate(range_threshold_lin):
    y_pred = (model.predict_proba(X_test)[:,1] >= thr) + 0.
    vector_precision[i] = metrics.precision_recall_fscore_support(y_test, y_pred)[0][1]
    vector_recall[i] =  metrics.precision_recall_fscore_support(y_test, y_pred)[1][1]
  sns.set()
  plt.plot(vector_precision, vector_recall)    
  plt.xlabel("Precision")
  plt.ylabel("Recall")
  plt.title("Tradeoff Precision-Recall Label 1");


# Plota o gráfico de features importance
def Plotar_feature_importance(model, X_test):
  fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': X_test.columns})
  fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
  fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 10), legend=None)
  plt.title('CatBoost - Feature Importance')
  plt.ylabel('Features')
  plt.xlabel('Importance')
  plt.show();

def diffdate(x,y):
    return ((x-y) / np.timedelta64(1, 'D')).astype(float)


# Simplifica copiar para o excel
def ex(self, n = None):
    if n is None:
        self.to_clipboard(excel = True);
    else:
        self.head(n).to_clipboard(excel = True);

setattr(pd.DataFrame, "ex", ex);


def count_stats(serie,log = False):
    n = len(serie)
    n_missing = sum(np.isnan(serie))
    print(f"Missing values: {n_missing}/{n} ({n_missing/n*100:.1f}%)\n")
    print(f"mean value: {np.nanmean(serie)}")
    print(f"minimum value: {serie.min()}")
    print(f"quantile 25%: {np.nanpercentile(serie,25)}")
    print(f"median value: {np.nanmedian(serie)}")
    print(f"quantile 75%: {np.nanpercentile(serie,75)}")
    print(f"maximum value: {serie.max()}")
    # plt.hist(serie, bins=10);


def count_stats_cat(serie, plot = False):
    n = len(serie)
    n_missing = sum(serie.isna())
    print(f"Missing values: {n_missing}/{n} ({n_missing/n*100:.1f}%)\n")
    if plot:
        serie.value_counts()[0:10].plot(kind = "barh")
        plt.show()
    else:
        print(serie.value_counts())





# Finds proportion of each category in a column which has the target
def proportion_target_categorical(df, col, name_target = 'Target', plot_with = 'plotly'):
    tmp = df.groupby(col)[name_target].mean().reset_index()
    if plot_with == 'plotly':
        fig = px.bar(tmp, x = col, y = name_target, height= 300)
        fig.show();
    else:
        plt.bar(tmp[col], tmp[name_target]);


# Finds proportion of the target in a column which is numerical by agregating in bins
def proportion_target_numeric(df, col, name_target = 'Target', plot_with = 'plotly'):
    tmp = df.groupby(col)[name_target].mean().reset_index()
    if plot_with == 'plotly':
        fig = px.bar(tmp, x = col, y = name_target, height= 300)
        fig.show();
    else:
        plt.bar(tmp[col], tmp[name_target]);


def concorrentes_subfonte(fonte):
    index_fonte = (df['Fonte'] == fonte)
    subfontes = df.loc[index_fonte, 'Subfonte'].unique()
    for subfonte in subfontes: # subfonte = subfontes[1]
        index_subfonte = (df['Subfonte'] == subfonte) & (df['Fonte'] == fonte)
        if sum(index_subfonte) > 100:
            df.loc[index_subfonte, 'Concorrente'].value_counts()[0:7].plot(kind = "barh", title = f"{subfonte}", figsize=(16,8))
            plt.show();



###################################################################
# Tira estatísticas
##################################################################

def concorrentes_fonte(fonte):
    index_fonte = (df['Fonte'] == fonte)
    df.loc[index_fonte, 'Concorrente'].value_counts()[0:10].plot(kind = "barh", figsize=(16,8))
    plt.show();


def count_stats_cat(serie, plot = False):
    n = len(serie)
    n_missing = sum(serie.isna())
    print(f"Missing values: {n_missing}/{n} ({n_missing/n*100:.1f}%)\n")
    if plot:
        serie.value_counts()[0:10].plot(kind = "barh")
        plt.show()
    else:
        print(serie.value_counts())




# Takes a pd.DataFrame table and adds a final column named 'Total'
# If normalize == True, then divides each number by the total, so it is represented as a proportion
def add_total(self, normalize = False):
      total = self.apply(sum,axis = 1); total.columns = ['Total']
      if normalize:
            self = pd.concat([self.div(total, axis = 0), total], axis = 1)
      else:
            self = pd.concat([self, total], axis = 1)
      self.rename(columns = {0:'Total'}, inplace = True)
      return self;


setattr(pd.DataFrame, 'add_total', add_total)

# an extension of combine_first where we specify the column to merge
# it is very slow
def combine_merge(self, df, on = None):
# left = pd.DataFrame
# >>> left       >>> right
#    a  b   c       a  c   d 
# 0  1  4   9    0  1  7  13
# 1  2  5  10    1  2  8  14
# 2  3  6  11    2  3  9  15
# 3  4  7  12    

# >>> left.combine_merge(right,on='a')
#    a  b  c   d
# 0  1  4  7  13
# 1  2  5  8  14
# 2  3  6  9  15
# 3  
    if on is None:
          columns_a = self.columns
          columns_b = df.columns 
          on = [i for i in columns_a if i in columns_b]
    return self.set_index(on).combine_first(df.set_index(on)).reset_index();


setattr(pd.DataFrame, 'combine_merge', combine_merge)



# def foo(self): # Have to add self since this will become a method
#     print('hello world!')
# setattr(DataFrame, 'to_pdcsv', to_pdcsv)


def rmerge(left,right,**kwargs):
    """
    Taken from: https://gist.github.com/mlgill/11334821
    
    Perform a merge using pandas with optional removal of overlapping
    column names not associated with the join. 
    
    Though I suspect this does not adhere to the spirit of pandas merge 
    command, I find it useful because re-executing IPython notebook cells 
    containing a merge command does not result in the replacement of existing
    columns if the name of the resulting DataFrame is the same as one of the
    two merged DataFrames, i.e. data = pa.merge(data,new_dataframe). I prefer
    this command over pandas df.combine_first() method because it has more
    flexible join options.
    
    The column removal is controlled by the 'replace' flag which is 
    'left' (default) or 'right' to remove overlapping columns in either the 
    left or right DataFrame. If 'replace' is set to None, the default
    pandas behavior will be used. All other parameters are the same 
    as pandas merge command.
    
    Examples
    --------
    >>> left       >>> right
       a  b   c       a  c   d 
    0  1  4   9    0  1  7  13
    1  2  5  10    1  2  8  14
    2  3  6  11    2  3  9  15
    3  4  7  12    
    
    >>> rmerge(left,right,on='a')
       a  b  c   d
    0  1  4  7  13
    1  2  5  8  14
    2  3  6  9  15
    >>> rmerge(left,right,on='a',how='left')
       a  b   c   d
    0  1  4   7  13
    1  2  5   8  14
    2  3  6   9  15
    3  4  7 NaN NaN
    >>> rmerge(left,right,on='a',how='left',replace='right')
       a  b   c   d
    0  1  4   9  13
    1  2  5  10  14
    2  3  6  11  15
    3  4  7  12 NaN
    
    >>> rmerge(left,right,on='a',how='left',replace=None)
       a  b  c_x  c_y   d
    0  1  4    9    7  13
    1  2  5   10    8  14
    2  3  6   11    9  15
    3  4  7   12  NaN NaN
    """

    # Function to flatten lists from http://rosettacode.org/wiki/Flatten_a_list#Python
    def flatten(lst):
        return sum( ([x] if not isinstance(x, list) else flatten(x) for x in lst), [] )
    
    # Set default for removing overlapping columns in "left" to be true
    myargs = {'replace':'left'}
    myargs.update(kwargs)
    
    # Remove the replace key from the argument dict to be sent to
    # pandas merge command
    kwargs = {k:v for k,v in myargs.items() if k is not 'replace'}
    
    if myargs['replace'] is not None:
        # Generate a list of overlapping column names not associated with the join
        skipcols = set(flatten([v for k, v in myargs.items() if k in ['on','left_on','right_on']]))
        leftcols = set(left.columns)
        rightcols = set(right.columns)
        dropcols = list((leftcols & rightcols).difference(skipcols))
        
        # Remove the overlapping column names from the appropriate DataFrame
        if myargs['replace'].lower() == 'left':
            left = left.copy().drop(dropcols,axis=1)
        elif myargs['replace'].lower() == 'right':
            right = right.copy().drop(dropcols,axis=1)
        
    df = pd.merge(left,right,**kwargs)
    
    return df




def Value_Counts(self, subset = '', dropna = True):
    cols = self.columns
    if len(subset) > 0:
        cols = subset
    for col in cols:
        print(f"\n{col}\n================================================")
        print(self[col].value_counts(dropna = dropna));


setattr(pd.DataFrame, 'Value_Counts', Value_Counts)


def perc_miss(self, plot = False):
    if plot:
        df.perc_miss().plot(kind = "barh", figsize = (14,11), title = "Non-null values in each column")
    else:
        return self.apply(lambda x: sum(x.notna())/len(x));


setattr(pd.DataFrame, 'perc_miss', perc_miss)
