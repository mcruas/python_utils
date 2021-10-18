# =============================================
# = Usuais
import os; os.chdir(path)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import glob; glob.glob("$HOME/*.csv") # caminho completo de todos *.csv
pasta_arquivo = os.path.join(pasta_raiz, pasta_filho, 'arquivo1.csv')


# _ Adicionar casos numa lista
case_list = []
for entry in entries_list:
    case = {'key1': entry[0], 'key2': entry[1], 'key3':entry[2] }
    case_list.append(case)
pd.concat(case_list) # transforma em Dataframe

# Arredondamentos
df[f"Churn_Risk_{alpha:.2f}"] = escores.round(4)

# ==============================================
# = Regular expressions
import re
s = "[MIGRAÇÃO] [CONCENTRA MODALIDADE] OPERAÇÃO RA"
s = 'ABS GQFS as[B]w'
re.findall(r"(?<=\[MIGRAÇÃO\] \[).*(?=\])", s)

df["alarmes"].str.replace(r'(\[concentra_modalidade\])+', '[concentra_modalidade]')

# ===============================================
# ==== dfply
from dfply import * 
(df_trx_day2 >> mask(X.SegmentoSmart == seg, X.SalesStructureNameLevel5 == polo) >>
                select(X.FullDate, X.Transactions))

# ===============================================
# ====== Plots ==================================
# ===============================================


########### PANDAS

#_ Cria um histograma que contém divisões por grupo
def hist_groups(df, column_hist, group, bins = 100):
  df.pivot(columns = column_hist, values = group).plot.hist(bins = bins);

# df.pivot(columns="Outcome", values="Glucose").plot.hist(bins=100)


# _ Define distribuição acumulada
def acumulada(coluna):
  coluna.plot(kind='hist',density=True,cumulative=-1,bins=1000,grid=True);

# Gráfico de frequência
df['class'].value_counts().plot('barh') #horizontal bar plot


# Salva pandas
fig = missing.plot(figsize=(16,12), kind = "barh").get_figure()


########## SEABORN 

# _ Seaborn
sns.set_style("whitegrid")

# _ Distribuição Acumulada
kwargs = {'cumulative': True}
plt.figure(figsize=(10,6))
ax = sns.distplot(df['coluna'], hist_kws=kwargs, kde_kws=kwargs, label = "blau")
ax.set(xlabel='X', ylabel='Probabilidade acumulada')
ax.legend() # cria legenda com as séries de nome dos labels
plt.show() # necessário após execução de for



# _ MELT de wide para long com objetivo de plot
df_plot = df.melt(id_vars=['reference_date']) # transforma todas as variáveis de coluna em variável



######### PLOTLY

import plotly.express as px
# px.density_contour(data_frame = df_plot,x = 'DiasSemTrx', y = 'churn_risk', marginal_x='histogram', marginal_y='histogram')
px.density_heatmap(data_frame = df_plot,x = 'DiasSemTrx', y = 'churn_risk', \
                   marginal_x='histogram', marginal_y='histogram', histnorm='density', nbinsx=20, nbinsy= 20)




# _ Plot de bar (ou line ou scatter)
fig = px.bar(df.loc[df['D'] == d, ['ReferenceDate','#']],'ReferenceDate', '#', title  = f"D = {d}",
  )
fig.update_layout(
    width=1500,
    height=300,
    xaxis_title="x Axis Title",
    margin=dict(
      l=0,
      r=0,
      b=10,
      t=20,
      pad=4
    )
)
fig.show() # colocar caso dentro de um loop

# _ Plot de linhas
px.line(df_plot, x = 'reference_date', y = 'value', color = 'variable', title = "Quantis dos Scores de Churn")



# ================================================================
# ======================= Pandas =================================
# ================================================================
pd.set_option('display.max_rows', 200)

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_columns', None)


## tira os caracteres nao ascii dos nomes das colunas do df
# import re
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


df['Tag_incidente'] =    ( 
        df['Tipo_incidente']
        .loc[df['CreatedDate'].dt.year == 2019] # Pega apenas o ano de 2019
    )
    
df['coluna'] = df['coluna'].astype('str') # Converter int para string

#_ strings
    .str.extract(r'(\[.*\])')[0] # pega apenas strings entre [ e ]
    .str[1:] # remove primeiro caracter
    .str[:-1] # remove último caracter
    .str.lower()  # coloca em letras minúsculas 
    # 3 próximas linhas para remover acentos
    .str.normalize('NFKD')
    .str.encode('ascii', errors='ignore')
    .str.decode('utf-8')
    .replace(depara_dict) # substitui as palavras conforme estão no dicionário depara_dict


# _ agregação
table = pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'], aggfunc=np.sum)

# Transformação de série diária em semanal, agregando como soma no último dia do intervalo da semana
df_week = (df_d.groupby(pd.Grouper(key='ReferenceDate', freq='W-MON'))['#']
      .sum()
      .reset_index()
      .sort_values('ReferenceDate')
    )


## Melhor forma de agregação, via pd.NamedAgg ou tuplas. 
def quantile_01(x):  return np.quantile(x,0.1) # ainda tem bug e lambda dá pau.
def quantile_09(x):  return np.quantile(x,0.9)

df = (df
     .groupby(["col2", "col3","col4", "col5"])
     .agg(
          TPV_Q01 = pd.NamedAgg('TPV', quantile_01),
          TPV_medio = ('TPV', 'mean'),
          TPV_Q09 = pd.NamedAgg('TPV', quantile_09),
          count = ('Id_Merge', 'count'),
          )
     .reset_index()
)

# tira outliers baseado em contato
df_loess.clip(0, df_loess.quantile(0.95), axis=1)



## Cria ocorrencias de value_counts como colunas
df.groupby('name')['activity'].value_counts(normalize = True).unstack().fillna(0)


## Reindexa para ter todos os valores 
datas = pd.date_range(calculo_trx_data_min, calculo_trx_data_max, freq='M')
cnpjs = df_trx_raw.query("Data in @datas")['CNPJ_CPF'].unique()
new_index = pd.MultiIndex.from_product([cnpjs,datas], names=['CNPJ_CPF', 'Data'])
df_trx_reindex = df_trx_raw[['Data', 'CNPJ_CPF', 'TPV']].set_index(['CNPJ_CPF', 'Data']).reindex(new_index, fill_value = 0).reset_index()



# _ I/O
pd.read_excel("planilha.xlsx")
pd.read_csv("file.csv")

######################### DATES ######################
# Dates manipulation
# Transforma a data para o último dia do mês
from pandas.tseries.offsets import MonthEnd
df['Data'] = pd.to_datetime(df['Data'], format="%Y%m") + MonthEnd(0)

# Seleciona datas dentro de um intervalo
in_range_df = df[df["date"].isin(pd.date_range("2017-01-15", "2017-01-20"))]

# Cria Range de datas
pd.date_range('2000-1-1', periods=200, freq='D')

# Data de hoje
pd.to_datetime("today")

# Muda a frequencia
df_m_menos_rx = (df
      .groupby(pd.Grouper(key='Data', freq='M')) # freq = W (semana)
      .count()
      .reset_index(name = 'count')
      .sort_values(['Data', 'Concorrente'])
    )

# Plota a quantidade de ocorrências semanais
df_inc_trx.groupby(pd.Grouper(key='Data', freq='W')).size().plot() # freq = W (semana)

# Coloca em formato Ano-semana
df_portal['Data'].dt.strftime('%Y-%V') 

# Teste
from pandas import util
# qdo deprecatar import pandas.util.testing
df1= util.testing.makeDataFrame().reset_index(drop=True).reset_index()
df2= util.testing.makeDataFrame().reset_index(drop=True).reset_index()
# df1['A'] = [i for i in range(30)]
df1.columns = ['index', 'k', 'f','g','h']
df1.merge(df2, how='left', on='index')


# ===========================================================================
# ============= Dictionary ==========================================

# Iterating over values 
for state, capital in statesAndCapitals.items(): 
    print(state, ":", capital) 


import copy
d = { ... }
d2 = copy.deepcopy(d)
# ===========================================================================
# ============= Files ==========================================


df_list = []
for entry in BASE_SAZONALIDADE.iterdir():
    if (entry.suffix == '.csv'):
            df1 = pd.read_csv(entry).melt('semana', var_name = "SS5", value_name = "sazonalidade")
            df1['MacroClassificacao'] = entry.name.split('.')[0]
            df_list.append(df1)                  
df_list1 = pd.concat(df_list) 

# ===============================================================================================
# ===============================================================================================
# ====================================== Tratamento Stone =======================================
# ===============================================================================================
# ===============================================================================================

# = Stonecodes: filtra apenas os que possuem 9 caracteres alfanuméricos e diferente de 999999999

    nome_col_stonecode = 'stonecode'
    df[nome_col_stonecode] = pd.to_numeric(df[nome_col_stonecode], errors = "coerce") # converte stonecode para numérico
    df = df[
        (df[nome_col_stonecode] != 999999999) &
         (df[nome_col_stonecode].notna())
    ]




# =====================================================================================
# ================================= Conexão Python DW =================================
# =====================================================================================




# =====================================================================================
# =============================================== Google Colab =================================
# =====================================================================================

from google.colab import drive
drive.mount('/drive/', force_remount=True)

import pandas as pd
import pyodbc
import os

%cd 'drive/My Drive/Gestão do Ciclo de Vida/6. Data Science/Churn - PUC/'

# Não exibir saída na celula
%%capture



# =====================================================================================
# =============================================== Pandas Profiling =================================
import pandas as pd
import pandas_profiling

profile = df.profile_report(title='Pandas Profiling Report')
profile.to_file(outputfile="Titanic data profiling.html")


# =====================================================================================
# =============================================== Jinja ===============================
from jinja2 import Template

name = input("Enter your name: ")

tm = Template("Hello {{ name }}")
msg = tm.render(name=name)

# =====================================================================================
# =============================================== Sweetviz =================================

# importing sweetviz
import sweetviz as sv
#analyzing the dataset
advert_report = sv.analyze(df, target_feat='label')
#display the report
advert_report.show_html('tmp.html')

# ====================================================================================
# ==================== Utilidades =================================================

# checa o sistema operacional
import platform 
platform.system() # 'Windows' ou 'Linux' 

# variáveis de ambiente
# https://able.bio/rhett/how-to-set-and-get-environment-variables-in-python--274rgt5
# Set environment variables
os.environ['API_USER'] = 'username'
os.environ['API_PASSWORD'] = 'secret'
# Get environment variables
USER = os.getenv('API_USER')
PASSWORD = os.environ.get('API_PASSWORD')

import importlib.util
spec = importlib.util.spec_from_file_location("module.name", "/path/to/file.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
foo.MyClass()