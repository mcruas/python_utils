


# Pegar resultados de tasks
with Flow('flat map') as f:
    params = make_parameters(model_specs, dates)
    results = apply_func.map(params)
    print(results)

states = f.run()
states.result[f.get_tasks()[1]].result

# Colocar múltiplos hiperparametros em .map (aceita dicionário)
@task
def make_parameters(model_specs, dates):
    def make_list_of_dict(d):
        return [{key: d[key]} for key in d.keys()]

    def convert_dicts(*args):
        lista = []
        for arg in args:
            if isinstance(arg, dict):
                # print('a')
                lista.append(make_list_of_dict(arg))
            else:
                lista.append(arg)
        return lista

    return list(itertools.product(*convert_dicts(model_specs, dates)))


def original_func(par1, par2):
    return par1 + par2 


@task(log_stdout=True)
def apply_func(args):
    par1, par2 = args
    out = original_func(model, date)
    return out




# Ler do bucket
@task
def read_bucket(filepath):
    from prefect.engine.serializers import PandasSerializer
    from prefect.tasks.gcp.storage import GCSDownload
    file_to_download = GCSDownload(
        bucket='tech-external-stoneflow',
        blob=filepath,
        project='s3l9orx36bur6pazk1unzl1z91pvwn'
    ).run()
    return PandasSerializer(file_type='csv') \
        .deserialize(file_to_download) \
        .drop(columns=['Unnamed: 0'], errors='ignore')



##################################
# Jinja 2
{% if env == 'nonprd' %}
TOP 10
{% endif %}

from jinja2 import Template

name = input("Enter your name: ")

tm = Template("Hello {{ name }}")
msg = tm.render(name=name)


##################################
# PrefectStone


### fact-rolling possui a maioria 
# -*- coding: utf-8 -*-
import pandas as pd
from prefect import task, Parameter
from prefectstone.tasks.sql_server import DW
from prefectstone.tasks.gcp import BigQueryTask
from prefectstone.utils import query_task
from prefectstone import StoneFlow
pd.set_option("max_columns", 50)


@query_task(filename='query_trx.sql')
def pegar_dados_trx(reference_date, env):
    bqct = DW()
    return bqct.run()


@task(log_stdout=True)
def df_on_format_datalake(df, reference_date=None):
    cols = ["rav_28d", "voucher_28d", "stone_account_cashin_28d",
            "stone_account_cashout_28d"]
    df[cols] = None
    df['reference_date'] = reference_date
    print(df.head(5))
    return df

@task
def send_to_datalake(df, env='nonprd'):
    res = BigQueryTask(env=env, df=df, event='fact-rolling')
    if res.run():
        return True
    return False


@task
def generate_date(date_parameter):
    if date_parameter == 'yesterday':
        return (pd.Timestamp.today() + pd.DateOffset(-1)).strftime('%Y-%m-%d')
    if date_parameter == 'today':
        return pd.Timestamp.today().strftime('%Y-%m-%d')
    return date_parameter


# %% Define Flows
with StoneFlow(name="fact-rolling", project="etl") as flow:
    reference_date = Parameter('reference_date', default='yesterday')
    ref = generate_date(reference_date)
    env = Parameter('env', default='nonprd')
    df = pegar_dados_trx(reference_date=ref, env=env)
    df = df_on_format_datalake(df, reference_date=ref)
    res = send_to_datalake(df=df, env=env)

# if __name__ == '__main__':
#     flow.run()

#%% Commands

# pip uninstall prefectstone
# pip install git+https://git@github.com/stone-payments/economic-research.git@develop
# prefectstone register flow -n fact-rolling -p etl -l agent_2