########################################################################
############################ PDCSV #####################################

import pandas as pd
import os
import matplotlib.pyplot as plt

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

# def foo(self): # Have to add self since this will become a method
#     print('hello world!')
# setattr(DataFrame, 'to_pdcsv', to_pdcsv)

# Cria um método para escrita em arquivo .pdcsv
def to_pdcsv(self, base, path = '', index=False, args={}):
    full_path = os.path.join(base, path)
    if index:
        # rename index
        index_cols = list(df.index.names)
        for i in range(0, len(index_cols)):
            if pd.isna(index_cols[i]):
                index_cols[i] = f"index{i}"
            else:
                pass
        self.index.names = index_cols
        self = self.reset_index()
        dtypes[index_cols] = dtypes[index_cols] + ":index"
    else:
        pass

    dtypes = self.dtypes.to_frame().transpose().astype(str)

    return pd.concat([dtypes, self], axis=0).to_csv(full_path, index=False, **args)

setattr(pd.DataFrame, 'to_pdcsv', to_pdcsv)


def read_pdcsv(path, args={}):
    # Read types first line of csv
    dtypes = pd.read_csv(path, nrows=1).iloc[0]

    # set dtype and parse_dates vector from dtypes
    parse_dates = list(dtypes[dtypes.str.contains("date")].index.values)
    index_col = list(dtypes[dtypes.str.endswith(":index")].index.values)
    dtypes = dtypes.str.split(":index").str[0]

    dtype = dtypes[~dtypes.index.isin(parse_dates)].to_dict()

    # Read the rest of the lines with the types from above
    if len(index_col) == 0:
        return pd.read_csv(
            path, dtype=dtype, parse_dates=parse_dates, skiprows=[1], **args
        )
    else:
        return pd.read_csv(
            path, dtype=dtype, parse_dates=parse_dates, skiprows=[1], **args
        ).set_index(index_col)


def list_last_files(base, path = '', num_last = 1, ext = ".pdcsv"):
    # Essa função pega o caminho dos últimos arquivos que foram gerados pelo processo em questão
    full_path = os.path.join(base, path)
    filename = pd.Series(os.listdir(full_path)).sort_values()
    filename = filename[filename.str.endswith(ext)].copy()
    lista = filename.iloc[-num_last:].values 
    return lista;


def get_latest_n_df(base, path = '', args = {}, num_last = 1, ext = ".pdcsv"):
# Uma função para pegar uma sequência das últimas bases de dados
    file_list = list_last_files(base, path, num_last, ext = ext)
    df_tmp = []
    for filename in file_list: # filename = file_list[1]
        file_path = os.path.join(base, path, filename)
        if ext == ".pdcsv":
            df_tmp.append(read_pdcsv(file_path, args = args))
        elif ext == ".csv":
            df_tmp.append(pd.read_csv(file_path, **args))
        elif ext == ".xlsx":
            df_tmp.append(read_excel(file_path, engine="openpyxl"))
        else: 
            print("Invalid extension")
        df_tmp = pd.concat(df_tmp, ignore_index= True, sort = True)
        print(f"Read file: {filename}\nDimensions: {df_tmp.shape}")
    return df_tmp


def count_stats_cat(serie, plot = False):
    n = len(serie)
    n_missing = sum(serie.isna())
    print(f"Missing values: {n_missing}/{n} ({n_missing/n*100:.1f}%)\n")
    if plot:
        serie.value_counts()[0:10].plot(kind = "barh")
        plt.show()
    else:
        print(serie.value_counts())


def path_fig(file_name):
  fig.savefig(os.path.join(DRIVE_BASES_STATISTICS_PATH, 'Frequencia_missing.jpg'))