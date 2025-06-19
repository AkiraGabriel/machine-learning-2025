# %%
import pandas as pd

df = pd.read_parquet("data/dados_clones.parquet")
df.head()
# %%
features = ["Massa(em kilos)", "General Jedi encarregado", "Estatura(cm)", "Distância Ombro a ombro", "Tamanho do crânio", "Tamanho dos pés", "Tempo de existência(em meses)"]
target = df['Status ']
# %%
df.columns
# %%
print(
    df['Status '].unique(),
    df['General Jedi encarregado'].unique(),
    df['Distância Ombro a ombro'].unique(),
    df['Tamanho do crânio'].unique(),
    df['Tamanho dos pés'].unique(),
)
# %%
df = df.replace({
    'Defeituoso':0,
    'Apto':1,
    'Tipo 1':1,
    'Tipo 2':2,
    'Tipo 3':3,
    'Tipo 4':4,
    'Tipo 5':5,
    'Yoda':0,
    'Shaak Ti':1,
    'Obi-Wan Kenobi':2,
    'Aayla Secura':3,
    'Mace Windu':4
})

# %%
df['General Jedi encarregado'].unique()
# %%
print(df['General Jedi encarregado'].unique())
print(df.columns)
# %%
y = df['Status ']
# %%
X = df[features]
y =  target
# %%
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# %%
model = tree.DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# %%
plt.figure(dpi = 400)

# %%
proba = model.predict(([[1,1,1,1,1,1,1]]))[0]
# %%
tree.plot_tree(model, max_depth=3,
               feature_names= features,
               class_names=model.classes_,
               filled=True)
# %%
pd.Series(proba, index=model.classes_)
# %%
tree.plot_tree(model, max_depth=4,
               feature_names= features,
               class_names=model.classes_,
               filled=True)

# %%
features.pop(1)
X = df[features]
# %%
model = tree.DecisionTreeClassifier(random_state=42)
model.fit(X, y)
# %%
plt.figure(dpi = 1000)
tree.plot_tree(model, max_depth=4,
               feature_names= features,
               class_names=model.classes_,
               filled=True)
# %%
features =["Massa(em kilos)", "Estatura(cm)"]

# %%
df['Status '] = df['Status '].replace({0:'Defeituoso', 1:'Apto'})
# %%
df.groupby('Status ')[features].mean()
