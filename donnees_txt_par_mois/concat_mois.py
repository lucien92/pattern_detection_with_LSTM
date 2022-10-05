import pandas as pd
import csv

#path to the txt file

path_to_gamma = '/home/lucien/Documents/projet-data-science-pollution-master/donnees_txt_par_mois/2015_gamma.txt'
path_to_HYGR = '/home/lucien/Documents/projet-data-science-pollution-master/donnees_txt_par_mois/2015_months_HYGR.txt'
path_to_PATM = '/home/lucien/Documents/projet-data-science-pollution-master/donnees_txt_par_mois/2015_months_PATM.txt'
path_to_TEMP = '/home/lucien/Documents/projet-data-science-pollution-master/donnees_txt_par_mois/2015_months_TEMP.txt'

#On souhaite transformer ces fichiers txt en un fichier à une colonne chacun

#On ouvre les fichiers txt
f_gamma = open(path_to_gamma, 'r')
f_HYGR = open(path_to_HYGR, 'r')
f_PATM = open(path_to_PATM, 'r')
f_TEMP = open(path_to_TEMP, 'r')

#On crée les listes qui vont contenir les données
gamma = []
HYGR = []
PATM = []
TEMP = []

#On parcourt les fichiers txt colonne par colonne et on ajoute les données dans les listes et pour cela on les convertit en dataframe puis on les transforme en liste
df_gamma = pd.read_csv(f_gamma, sep=',')
for mots in df_gamma['fevrier']:
    gamma.append(mots)
for mots in df_gamma['avril']:
    gamma.append(mots)
for mots in df_gamma['juin']:
    gamma.append(mots)
for mots in df_gamma['octobre']:
    gamma.append(mots)

df_HYGR = pd.read_csv(f_HYGR, sep=',')
for mots in df_HYGR['fevrier']:
    HYGR.append(mots)
for mots in df_HYGR['avril']:
    HYGR.append(mots)
for mots in df_HYGR['juin']:
    HYGR.append(mots)
for mots in df_HYGR['octobre']:
    HYGR.append(mots)

df_PATM = pd.read_csv(f_PATM, sep=',')
for mots in df_PATM['fevrier']:
    PATM.append(mots)
for mots in df_PATM['avril']:
    PATM.append(mots)
for mots in df_PATM['juin']:
    PATM.append(mots)
for mots in df_PATM['octobre']:
    PATM.append(mots)

df_TEMP = pd.read_csv(f_TEMP, sep=',')
for mots in df_TEMP['fevrier']:
    TEMP.append(mots)
for mots in df_TEMP['avril']:
    TEMP.append(mots)
for mots in df_TEMP['juin']:
    TEMP.append(mots)
for mots in df_TEMP['octobre']:
    TEMP.append(mots)


#On ferme les fichiers txt
f_gamma.close()
f_HYGR.close()
f_PATM.close()
f_TEMP.close()

#on écrit le chemin des fichiers txt qui vont contenir les données

path_to_gamma_concat = '/home/lucien/Documents/projet-data-science-pollution-master/donnees_txt_par_mois/2015_gamma_concat.txt'
path_to_HYGR_concat = '/home/lucien/Documents/projet-data-science-pollution-master/donnees_txt_par_mois/2015_months_HYGR_concat.txt'
path_to_PATM_concat = '/home/lucien/Documents/projet-data-science-pollution-master/donnees_txt_par_mois/2015_months_PATM_concat.txt'
path_to_TEMP_concat = '/home/lucien/Documents/projet-data-science-pollution-master/donnees_txt_par_mois/2015_months_TEMP_concat.txt'

#On crée les fichiers txt qui vont contenir les données

f_gamma = open(path_to_gamma_concat, 'w')
f_HYGR = open(path_to_HYGR_concat, 'w')
f_PATM = open(path_to_PATM_concat, 'w')
f_TEMP = open(path_to_TEMP_concat, 'w')

#on transforme les listes de float en list de string

gamma = [str(i) for i in gamma]
HYGR = [str(i) for i in HYGR]
PATM = [str(i) for i in PATM]
TEMP = [str(i) for i in TEMP]


#On écrit les données dans les fichiers txt
with open('/home/lucien/Documents/projet-data-science-pollution-master/donnees_txt_par_mois/2015_gamma_concat.txt', 'w', newline='') as f_gamma:
    spamwriter_gamma = csv.writer(f_gamma, delimiter=',')
    spamwriter_gamma.writerow(['Date', 'gamma'])
    i = 0
    for item in gamma:
        i += 1
        spamwriter_gamma.writerow([i, item])

#on veut générer une time series test moins longue extraite de gamma pour tester le LSTM

with open('/home/lucien/Documents/projet-data-science-pollution-master/donnees_txt_par_mois/2015_gamma_concat_test.txt', 'w', newline='') as f_gamma:
    spamwriter_gamma = csv.writer(f_gamma, delimiter=',')
    spamwriter_gamma.writerow(['Date', 'gamma'])
    i = 0
    for item in gamma:
        if i<100:
            i += 1
            spamwriter_gamma.writerow([i, item])


# with open('/home/lucien/Documents/projet-data-science-pollution-master/donnees_txt_par_mois/2015_months_HYGR_concat.txt', 'w', newline='') as f_HYGR:
#     spamwriter_HYGR = csv.writer(f_HYGR, delimiter=',')
#     spamwriter_HYGR.writerow(['HYGR'])
#     for item in HYGR:
#         spamwriter_HYGR.writerow([item])

# with open('/home/lucien/Documents/projet-data-science-pollution-master/donnees_txt_par_mois/2015_months_PATM_concat.txt', 'w', newline='') as f_PATM:
#     spamwriter_PATM = csv.writer(f_PATM, delimiter=',')
#     spamwriter_PATM.writerow(['PATM'])
#     for item in PATM:
#         spamwriter_PATM.writerow([item])

# with open('/home/lucien/Documents/projet-data-science-pollution-master/donnees_txt_par_mois/2015_months_TEMP_concat.txt', 'w', newline='') as f_TEMP:
#     spamwriter_TEMP = csv.writer(f_TEMP, delimiter=',')
#     spamwriter_TEMP.writerow(['TEMPS'])
#     for item in TEMP:
#         spamwriter_TEMP.writerow([item])



#On ferme les fichiers txt
f_gamma.close()
# f_HYGR.close()
# f_PATM.close()
# f_TEMP.close()
