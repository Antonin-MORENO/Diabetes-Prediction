import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("diabetes_prediction_dataset.csv")

print(df.head())
df.head()
df.info()

print(df.isna().sum()) # pas de valeur manquante 


#As we can see, the dataset is quit clean indeed there is no missing values. But we have so transformation to do, first
# for the smoking presence we have some values equal to no info 


#Since we have 35000 rows with no info we should replace it.
counts = df['smoking_history'].value_counts()
plt.figure(figsize=(8, 6))
counts.plot(kind='bar')
plt.title('Distribution of Smoking History Categories')
plt.show()

#commetn on peut le voir malgré qu'on est pas de ligne vide pour la fetaure fumeur on a 35000 observations pour lesquelles ont a pas d'infos étant donner que c'est 35% 
# des données on ne va pas remplacer no info par une supposition au risque de ce trmper et va quanb meme les encoder 

counts_gender = df['gender'].value_counts()
plt.figure(figsize=(8, 6))
counts_gender.plot(kind='bar')
plt.title('Distribution of Smoking History Categories')
plt.show()

#comme on peut le voir il y quelque valeur pour lequel le sexe est autre cette fois ci comment ca represente une petite proportion de notre dataset on va les remplacer
#par la majoriter donc femme puis encoder en binaire






