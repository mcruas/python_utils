#############################################################################
###################### ANALISE EXPLORATORIA #################################

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


# def foo(self): # Have to add self since this will become a method
#     print('hello world!')
# setattr(DataFrame, 'to_pdcsv', to_pdcsv)
