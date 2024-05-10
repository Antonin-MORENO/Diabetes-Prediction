import pandas as pd

df = pd.read_csv("diabetes_prediction_dataset.csv")

print(df.head())
df.head()
df.info()

print(df.isnull().sum()) # pas de valeur manquante 


#As we can see, the dataset is quit clean indeed there is no missing values. But we have so transformation to do, first
# for the smoking presence we have some values equal to no info 

no_info_count = (df['smoking_history'] == "current").sum()

print("Nombre de lignes avec 'No Info' :", no_info_count)

#Since we have 35000 rows with no info we should replace it.