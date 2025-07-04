# %%
import pandas as pd

url = 'https://docs.google.com/spreadsheets/d/1YQBQ3bu1TCmgrRch1gzW5O4Jgc8huzUSr7VUkxg0KIw/export?gid=283387421&format=csv'

df = pd.read_csv(url)
df.head()
# %%
df = df.replace({"Sim":1, "Não":0})
df.head()
# %%
columns_str = ['Como conheceu o Téo Me Why?',
                'Quantos cursos acompanhou do Téo Me Why?',
                'Estado que mora atualmente', 
                'Área de Formação',
                'Tempo que atua na área de dados', 
                'Posição da cadeira (senioridade)']

pd.get_dummies(df[columns_str])
# %%
