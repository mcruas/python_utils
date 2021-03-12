# =============================================
# = Usual commands to ploting

# ========================================================
# PANDAS PLOTS

# Choose backend
pd.options.plotting.backend = 'matplotlib' # 'plotly', 'hvplot', 'bokeh'

# Ploting with groupers by Month. 
tmp = df.groupby(pd.Grouper(key='Data', freq='M')).size()
tmp.index = tmp.index.to_period('M') 
g = tmp.plot(kind = "bar", figsize = (12,6), rot = 45)
g.set_xlabel("x label")
g.set_ylabel("y label")
g.yaxis.set_major_formatter(PercentFormatter(1))

# Multiple plots
# https://www.kite.com/python/answers/how-to-plot-pandas-dataframes-in-a-subplot-in-python
figure, axes = plt.subplots(1, 2)
df1.plot(ax=axes[0])
df2.plot(ax=axes[1])


# Saving by Month
fig = tmp.plot(kind = "bar", figsize = (12,6), rot = 45).get_figure()
fig.savefig('Frequencia_missing.jpg', bbox_inches='tight')

# ========================================================
# MATPLOTLIB
# Plots grids
df=pd.DataFrame(np.random.choice(list("abcd"), size=(100,20), p=[.4,.3,.2,.1]))
fig, axes =plt.subplots(5,4, figsize=(10,10), sharex=True)
axes = axes.flatten()
object_bol = df.dtypes == 'object'
for ax, catplot in zip(axes, df.dtypes[object_bol].index):
    sns.countplot(y=catplot, data=df, ax=ax, order=np.unique(df.values))
plt.tight_layout()  
plt.show()


# Boxplots construídos iterativamente
x = pd.DataFrame(np.random.randn(10, 10))
fig = plt.figure()
ax = plt.subplot(111)
n_plots = 6
for i in range(n_plots):
    ax.boxplot(x.iloc[:,i].values, positions = [i])

ax.set_xlim(-0.5, n_plots - 0.5)
plt.show()



# =================================================================
# SEABORN

sns.set_style("whitegrid")

#from  matplotlib.ticker import PercentFormatter
g = sns.catplot(x = 'Concorrente', y = 'Perc_Na_Master', 
	kind = 'bar', data = tmp, aspect = 3, height = 3)
# Mudar nome de labels labels
g.set_axis_labels("Concorrente","")
# Colocar eixos em percentagem
for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(PercentFormatter(1))
# Salvar figura ; Deve ir antes de plt.show() !!!
plt.savefig(os.path.join(REPORTS_PLOTS, 'Concorrentes_Master.pdf'), bbox_inches = 'tight')
plt.show()


# Mudar tamanho em plot normal
plt.figure(figsize=(20,5))
sns.boxplot(
    data=summary,
    x='Country',
    y='medal count',
    color='red')



def save_fig(nome):
    plt.savefig(REPORTS_PLOTS.joinpath(nome), bbox_inches = 'tight');
