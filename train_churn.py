# train_churn.py
# versao que funciona (NAO MEXER q quebra)
# ana - churn do trabalho
# TODO: organizar isso um dia qdo tiver tempo

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

# caminho do dataset (se rodar em outra maquina mudar aqui)
df = pd.read_csv("C:/Users/ana/Desktop/churn.csv")

print(df.shape)
print(df.head())
print("churn:", df["Churn"].value_counts())   # so pra ver

# tira o id que nao serve pra nada
df = df.drop("customerID", axis=1)

# TotalCharges vem como texto por algum motivo, forco pra numero
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(2200)   # media +- , ta ok

# feature nova que eu inventei (deu uma melhorada)
df["gasto_por_mes"] = df["TotalCharges"] / (df["tenure"] + 1)   # +1 pra nao dividir por zero

# limpa o resto dos nulos
df = df.dropna()

# converte tudo que eh texto pra numero
le = LabelEncoder()
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = le.fit_transform(df[c])

# alvo e features
y = df["Churn"]
X = df.drop("Churn", axis=1)

# normaliza as colunas grandes (li que ajuda o modelo)
X["MonthlyCharges"] = X["MonthlyCharges"] / 118.0
X["TotalCharges"] = X["TotalCharges"] / 8600.0
X["tenure"] = X["tenure"] / 72.0

# tentei gridsearch mas demorava muito, deixei fixo mesmo
# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(RandomForestClassifier(), {"n_estimators":[100,200,500]})
# grid.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("acuracia:", accuracy_score(y_test, pred))
# print(classification_report(y_test, pred))  # depois eu vejo isso

# acc = 0.79 ??? (ontem tinha dado 0.80, sei la)

# salva o modelo
pickle.dump(model, open("modelo_final_v3_ok.pkl", "wb"))
print("salvo!")
