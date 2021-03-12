


# Pegar resultados de tasks
with Flow('flat map') as f:
    params = make_parameters(model_specs, dates)
    results = apply_func.map(params)
    print(results)

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

