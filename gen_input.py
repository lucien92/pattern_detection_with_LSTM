#importations

import pandas as pd

#Maintenant qu'on a sélectionné nos motifs on va prendre une cinquantaine de séquences ne correspondant à aucun motifs mais faisant les mêmes tailles



#on prends les deux dataframe depuis leur csv
df_gamma = pd.read_csv("donnees_txt_par_mois/2015_gamma_concat.txt", sep=',')

df_motifs_gamma = pd.read_csv("/home/lucien/Documents/projet-data-science-pollution-master/data_for deep/motifs_gamma.csv", sep = ',')
print(df_motifs_gamma)

#on transforme nos base de données contenant des séquences de strings en séquences d'array

motif = df_motifs_gamma['motif']
print("len(motif):", len(motif))
cluster = df_motifs_gamma['cluster']

#on veut supprimer la première ligne de df
df_motifs_gamma = df_motifs_gamma.drop(df_motifs_gamma.index[0])
print(df_motifs_gamma) 

#Code pour transformer notre BD en un dataframe qui ne contient pas des une liste string mais des listes de float

print(type(motif[1]))   #avant on a des listes de string

for i in range(len(motif)):
    truc = motif[i]
    print(truc)
    #on souhaite que truc ne soit plus une liste de string mais une liste de float
    truc = truc.replace('[','')
    truc = truc.replace(']','')
    truc = truc.replace(' ','')
    truc = truc.split(',')
    truc = [float(i) for i in truc]
    motif[i+1] = truc


print(type(motif[1]))
print(type(motif[1][0]))

#on veut supprimer les lignes de df_gamma qui ont un l'index d'un nombre qui se trouve dans df_motifs_gamma

# df_gamma.drop(df_gamma.index[df_motifs_gamma['numéro_du_cluster']], inplace=True)

# print(df_gamma)







