#### Importations
import pandas as pd
import numpy as np
import stumpy_git as stumpy
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.patches import Rectangle
import datetime as dt
from scipy.signal import argrelmin
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
import matplotlib

## Partie Deep Learning: on fait le postulat que l'on recherche des motifs de toutes les tailles, même si l'un est imbriqué dans l'autre :
#Hypothèses de travail:
# -deux motifs imbriqués seront considérés comme différents motifs
# -les résultats donnés par la matrix profile seront considérés comme de confiance et constitueront les motifs annotés 


### Visualisation des données sur lesquelles on va travailler

#il faudrait rassembler les 4 mois dans un même fichier pdf

df_gamma = pd.read_csv("donnees_txt_par_mois/2015_gamma_concat.txt", sep=',')

# df_HYGR = pd.read_csv("donnees_txt_par_mois/2015_months_HYGR_concat.txt", sep=',')
# df_PATM = pd.read_csv("donnees_txt_par_mois/2015_months_PATM_concat.txt", sep=',')
# df_TEMP = pd.read_csv("donnees_txt_par_mois/2015_months_TEMP_concat.txt", sep=',')
#data_df = pd.DataFrame({'timestamp': np.arange(len(df["juin"])),"values":df["juin"]})

#### On boucle sur les différentes tailles de matrix profile

boo = True
N = 20 #nombre de minimum locaux que l'on recherche
A = [300,500,700,900, 1200] #tailles de motifs que l'on recherche

for m in A:
    #on clean note matrix profile
    mp = stumpy.stump(df_gamma['gamma'], m)
    data = df_gamma['gamma']
    mpp = mp.copy()
    mpp2 = pd.DataFrame(mpp)
    del mpp2 [2]
    del mpp2 [3]
    mpp3 = mpp2.copy()
    del mpp3[1]
    mpp3bis = np.array(mpp3)

    #Sélection des tous les minimaux locaux dont la valeur de la Matrix Profile est inférieur à une certaines limite 

    X_lim = min(mpp3bis)
    x_tuple = []
    while(len(x_tuple) < 4.5*N):

        x = argrelmin(mpp3bis,order=20) #prend min local et si deux min local on un écart de moins que order il prend le + petit (intéresant car sur un matrix profile de m=200 on parlera du même motif pour un pic et ses points autour de 100,150)
        x_tuple = []
        for ind in x[0]:
            if(mpp3bis[ind]<X_lim): #on prend les min locaux en desous de 8
                x_tuple.append((ind,mpp3bis[ind]))

        X_lim = 1.02*X_lim 

    x_tuple_sorted_by_value = sorted(x_tuple, key=lambda tup: tup[1])    #tri les minimum dans l'ordre croissant (les plus bas minimum d'abord)
    x_tuple_sorted_by_value = np.array(x_tuple,dtype=object)

    #Vérifie parmis tous ces minimaux locaux, qu'ils ne sont pas en double

    if(len(x_tuple_sorted_by_value) > 0):
        list_ind = np.array(x_tuple_sorted_by_value)[:,0]

        L=[] 
        for i in list_ind:
            test = True        
            for ind in L:
                if(abs(i-ind)<m):
                    test = False
                
            if(test):
                L.append(i)
            

            i = mpp2[1][i]
            test = True        
            for ind in L:
                if(abs(i-ind)<m):
                    test = False
                
            if(test):
                L.append(i)
    #Liste des N minimaux locaux à la plus petite valeur de Matrix Profile
    if(len(L) > N):
        L = L[0:N]

    L.sort()

    ## METHODE DES KMEANS POUR LES SERIES TEMPORELLES 
    ### Trouver le nombre de clusters optimal
    #on applique la méthode de la silhouette pour trouver le nombre de cluster optimal

    matplotlib.rcdefaults()
    W=[]
    for i in range(len(L)):
        if not(i in [9,12]):
            y = np.array(data[L[i]:L[i]+m])
            W.append((y - np.mean(y))/np.std(y)) #on normalise les données
            #W.append(y)#on ne les normalise pas

    scores = {}
    for n_clusters in range(2,len(W)):
        model_eucl=TimeSeriesKMeans(n_clusters, n_init=10)

        prediction_eucl = model_eucl.fit_predict(W)
        silhouette_avg = silhouette_score(W, prediction_eucl)
        scores[n_clusters] = silhouette_avg


    k_opt = [k for k, v in scores.items() if v == max(scores.values())][0]#donne le nombre de cluster optimal
    print("Nombre de clusters optimal : " + str(k_opt))
    model_eucl=TimeSeriesKMeans(k_opt, n_init=10) 

    prediction_eucl = model_eucl.fit_predict(W)

    print(prediction_eucl)

    def calculate_WSS(points, kmax):
        sse = []
        for k in range(1, kmax+1):
            kmeans = TimeSeriesKMeans(n_clusters = k, n_init=10).fit(points)
            centroids = kmeans.cluster_centers_
            pred_clusters = kmeans.predict(points)
            curr_sse = 0
            
            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(points)):
                curr_center = centroids[pred_clusters[i]]
                for j in range(m):
                    curr_sse += (points[i][j] - curr_center[j]) ** 2
                
            sse.append(curr_sse)

        return sse

    #avec ce code on peut écrire dans un csv la liste des cluster et le motif associé
    #il faudra faire attention car si on veut faire varier m il faut prendre en compte que certain motifs peuvent se chevaucher ou même être inclus l'un dans l'autre


    import csv

    #code pour écrire un csv propre qui ne revient pas à la ligne à chaque fois
    Z = []
    a = 0
    for i in W:
        i = list(i)
        Z.append(i)
        a +=1
    print("Le nombre de cluster pour cette taille de m est:", len(Z))
    with open('datamotifs.txt', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=" ")
        if boo == True:                       
            spamwriter.writerow(['motif' + ',' + 'cluster' + ',' + 'taille_du_motif' + ',' + 'numéro_du_cluster'])
            boo = False
        for i in range(len(W)):
            spamwriter.writerow([Z[i],',', prediction_eucl[i], ',', f"{m}", ',' ,f"{prediction_eucl[i]}_{m}"])

    import pandas as pd
    #on veut maintenant convertir ce csv en un dataframe

    df = pd.read_csv('datamotifs.txt', sep=',',header=None)
    df.columns = ['motif','cluster','taille_du_motif','numéro_du_cluster']


    motif = df['motif']
    cluster = df['cluster']
    #on veut supprimer la première ligne de df
    df = df.drop(df.index[0])
    print(df) #dataframe qui contient nos listes de motifs mais de type et les clusters auxquels ils appartiennent


    #Code pour transformer notre BD en un dataframe qui ne contient pas des une liste string mais des listes de float

    print(type(motif[1]))   #avant on a des listes de string

    for i in range(len(W)):
        truc = motif[i+1]
        #on souhaite que truc ne soit plus une liste de string mais une liste de float
        truc = truc.replace('[','')
        truc = truc.replace(']','')
        truc = truc.replace(' ','')
        truc = truc.split(',')
        truc = [float(i) for i in truc]
        motif[i+1] = truc

    print(type(motif[1]))
    print(type(motif[1][0])) #après on a des listes de float pour les motifs de toutes les tailles recherchées car on concatène la data à la fin


#on souhaite créer un csv contenant toutes les suites de nombres de tailles 300,500,700,900, 1200 qui ne sont pas de smotifs






#on souhaite maintenant noter tous ces motifs dans un fichier csv

df.to_csv("/home/lucien/Documents/projet-data-science-pollution-master/data_for deep/motifs_gamma.csv")