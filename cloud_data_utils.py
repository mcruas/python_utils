
import pandas as pd
import os
import matplotlib.pyplot as plt
# for xlsx requires 

########################################################################
############################ PDCSV #####################################

# Cria um método para escrita em arquivo .pdcsv
def to_pdcsv(self, base, path = '', index=False, auto_name = True, args={}):
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

# Atribui como método da classe DataFrame
setattr(pd.DataFrame, 'to_pdcsv', to_pdcsv)


def read_pdcsv(base, path = '', args={}):
    # Read types first line of csv
    full_path = os.path.join(base, path) if path != '' else base
    dtypes = pd.read_csv(full_path, nrows=1).iloc[0]

    # set dtype and parse_dates vector from dtypes
    parse_dates = list(dtypes[dtypes.str.contains("date")].index.values)
    index_col = list(dtypes[dtypes.str.endswith(":index")].index.values)
    dtypes = dtypes.str.split(":index").str[0]

    dtype = dtypes[~dtypes.index.isin(parse_dates)].to_dict()

    # Read the rest of the lines with the types from above
    if len(index_col) == 0:
        return pd.read_csv(
            full_path, dtype=dtype, parse_dates=parse_dates, skiprows=[1], **args
        )
    else:
        return pd.read_csv(
            full_path, dtype=dtype, parse_dates=parse_dates, skiprows=[1], **args
        ).set_index(index_col)


#################################################################
#### Leitura de arquivos na base

def list_last_files(base, path = '', num_last = 1, ext = ".pdcsv"):
    # Essa função pega o caminho dos últimos arquivos que foram gerados pelo processo em questão
    # Útil para quando se deseja carregar diversos arquivos mensais na mesma pasta
    full_path = os.path.join(base, path) if path != '' else base
    filename = pd.Series(os.listdir(full_path)).sort_values()
    filename = filename[filename.str.endswith(ext)].copy()
    lista = filename.iloc[-num_last:].values 
    return lista;


def get_latest_n_df(base, path = '', args = {}, num_last = 1, ext = ".pdcsv"):
## EXEMPLO: df = cd.get_latest_n_df(DRIVE_BASE, 'Base_TPV', args = {'limit' = 1000}, num_last = 10) 
## A função acima irá buscar os 10 arquivos mais recentes na pasta 'DRIVE_BASE/Base_TPV',
## pegar as primeiras 1000 linhas de cada um deles, concatenar e atribuir para df.
## o que tiver em args é passado como argumento para as funções de leitura
    file_list = list_last_files(base, path, num_last, ext = ext)
    full_path = os.path.join(base, path) if path != '' else base
    df_tmp = []
    for filename in file_list: # filename = file_list[1]
        file_path = os.path.join(full_path, filename)
        if ext == ".pdcsv":
            df_tmp.append(read_pdcsv(file_path, args = args))
        elif ext == ".csv":
            df_tmp.append(pd.read_csv(file_path, **args))
        elif ext == ".xlsx":
            df_tmp.append(pd.read_excel(file_path, engine="openpyxl"))
        else: 
            print("Invalid extension")
    df_tmp = pd.concat(df_tmp, ignore_index= True, sort = False)
    print(f"Read files: {file_list}\nDimensions: {df_tmp.shape}")
    return df_tmp


# def path_fig(file_name):
#   fig.savefig(os.path.join(DRIVE_BASES_STATISTICS_PATH, 'Frequencia_missing.jpg'))