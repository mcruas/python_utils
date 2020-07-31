'''
# Package imports
''' #########################################



# ! Uncomment when using Google Colab
# root_data = "/gdrive/"
!git clone https://github.com/mcruas/python_utils 
user_root = "/gdrive/"
import sys
sys.path.append('/content/python_utils')
import eda_utils as eda
import cloud_data_utils as cd
%cd /gdrive
'''
# Package imports
''' #########################################
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
import importlib

# Move to local libraries base path - $USER/Github
user_root = os.getenv('userprofile')
root_github = os.chdir(os.path.join(user_root,'Github'))


# ! Uncomment when using PC
root_data = "G:/"
# Local libraries imports
# Necessary to clone python_utils: !git clone https://github.com/mcruas/python_utils 
import python_utils.eda_utils as eda
import python_utils.cloud_data_utils as cd
# importlib.reload(eda)

# External files base path 
DRIVE_BASE_PATH = os.path.join(root_data,"Shared drives/darwin/darwin-mestro/")
DRIVE_BASE_MASTER_PATH = os.path.join(root_data,"Shared drives/darwin/base-concorrencia/Base_Master")
DRIVE_BASES_AUXILIARES_PATH = os.path.join(root_data,"Shared drives/darwin/base-concorrencia/Bases_Auxiliares")
DRIVE_BASES_STATISTICS_PATH = os.path.join(root_data,"Shared drives/darwin/base-concorrencia/Estatisticas")
DRIVE_BASES_INFO_DB = os.path.join(user_root,"My Drive/Pricing Stone Co/00. Bases/02. Infos DB")


# Reports folder
REPORTS_BASE = os.path.join(root_data,"My Drive/")

# Local file base_path
LOCAL_FILE_BASE = os.path.join(user_root, 'Local_Folder')

# Parameter settings
sns.set_style("whitegrid")
sns.set(font_scale=1.5) 
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


############################################
# Prepare data for analysis
df = cd.get_latest_n_df(DRIVE_BASE_MASTER_PATH, ext = ".pdcsv")





############################################
# Analysis



############################################
# 
df.to_pdcsv(LOCAL_FILE_BASE)