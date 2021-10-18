# adds a suffix or preffix to a list of strings
def get_stone_affs(**kwargs):
	pt = "gs://tech-external-stoneflow/common_data/stonedw-affiliation.csv"
	return (
		pd.read_csv(pt, **kwargs)
			.drop_duplicates(subset=['cpf_cnpj'], ignore_index=True)
			.drop(columns=['Unnamed: 0', 'affiliationkey', 'clientkey', 'stonecode'])
	) 
