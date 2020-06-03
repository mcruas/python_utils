#############################################################################
###################### ANALISE EXPLORATORIA #################################
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import pandas as pd
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


def Value_Counts(self, subset = '', dropna = True):
    cols = self.columns
    if len(subset) > 0:
        cols = subset
    for col in cols:
        print(f"\n{col}\n================================================")
        print(self[col].value_counts(dropna = dropna))


setattr(pd.DataFrame, 'Value_Counts', Value_Counts)


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



# def foo(self): # Have to add self since this will become a method
#     print('hello world!')
# setattr(DataFrame, 'to_pdcsv', to_pdcsv)
