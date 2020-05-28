##################################################
# Resume 



##################################################

# Package imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np

# Move to local libraries base path
user_root = os.getenv('userprofile')
root_github = os.chdir(os.path.join(user_root,'Github'))

# Local libraries imports
import python_utils.eda_utils as eda
import python_utils.cloud_data_utils as cd


# Select GoogleDrive or PC
root_data = "/gdrive/"
# root_data = "G:/"

# Data files base path 
DRIVE_BASE_PATH = os.path.join(root_data,"Shared drives/darwin/darwin-mestro/")
DRIVE_BASE_MASTER_PATH = os.path.join(root_data,"Shared drives/darwin/base-concorrencia/Base_Master")
DRIVE_BASES_AUXILIARES_PATH = os.path.join(root_data,"Shared drives/darwin/base-concorrencia/Bases_Auxiliares")
DRIVE_BASES_STATISTICS_PATH = os.path.join(root_data,"Shared drives/darwin/base-concorrencia/Estatisticas")

LOCAL_FILE_BASE = os.path.join(user_root, 'Bases_Locais')

# Parameter settings
sns.set_style("whitegrid")
sns.set(font_scale=1.5) 
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


############################################
# Leitura e preparação dos dados
df = rd.get_latest_n_df(DRIVE_BASE_MASTER_PATH, ext = ".pdcsv")





############################################
# Análise/Previsão


