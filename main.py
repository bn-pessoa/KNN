import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv("./Data_Train (2).csv", sep='\t')
X = df['MktCoupons', 'OriginAirportID', 'DestAirportID','ID_C'].values
y = df['MktFare'].values

vizinho = KNeighborsClassifier(n_neighbors=100)
vizinho.fit(X, y)

print(f"Preço predito: {vizinho.predict([[1.1]])}")
print(f"Probabilidade de X: {vizinho.predict_proba(X)}")
