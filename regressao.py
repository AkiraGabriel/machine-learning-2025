# %%
import pandas as pd
df = pd.read_excel('data/dados_cerveja_nota.xlsx')
# %%
from sklearn import linear_model

X = df[['cerveja']]
y = df['nota']

reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X,y)
# %%
a, b = reg.intercept_, reg.coef_[0]
print(a, b)