#System Libraries
import os
import sys

#ML Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from matplotlib import pyplot as plt
import seaborn as sns

#Time Library
import time

#Graphical interface User Libraries
from tkinter import *
from tkinter import font
from tkinter import ttk
from PIL import Image
from PIL import ImageTk              
from tkinter import messagebox
from tkinter.messagebox import *
from tkinter import filedialog
from subprocess import Popen, PIPE
import re

# On crée une fenêtre, racine de notre interface
fenetre = Tk()

fenetre.title("Fil Rouge ML Application")
fenetre.geometry("1350x690")    #("Largeur x Longeur")
fenetre.resizable(width=True, height=True)
fenetre.configure(background="grey")
#style = ttk.Style()
#style.configure("BW.TLabel", foreground="black", background="white")
#object_style.configure(troughcolor="gray", background="white")

##Photo Logo Application
SMS_img = Image.open("SMS_Alert.jpg")
SMS_img = SMS_img.resize((100, 100), Image.ANTIALIAS)
photo_SMS = ImageTk.PhotoImage(SMS_img)
photo_SMS_label = Label(fenetre, image = photo_SMS,background="grey")
photo_SMS_label.place(x=20, y=10)

##Photo Telecom Paris
Telecom_img = Image.open("Telecom_paris.jpg")
Telecom_img = Telecom_img.resize((60, 100), Image.ANTIALIAS)
photo_Telecom = ImageTk.PhotoImage(Telecom_img)
photo_Telecom_label = Label(fenetre, image = photo_Telecom,background="grey")
photo_Telecom_label.place(x=1260, y=10)

##Label N 1
label_1 = Label(fenetre, text="Charger les données :", background="grey", font = ("arial", 10, "bold"))
label_1.place(x=20, y=180)

##Label N 2
label_2 = Label(fenetre, text="",background="grey",  font = ("arial", 12))         #"Le fichier est chargé ... OK!
label_2.place(x=20, y=250)

##Creation du zone Texte
value = StringVar() 
value.set("Insérer le fichier (Dataset) ...")
entree = Entry(fenetre, textvariable = "string",  width=45)
entree.place(x=20, y=210)

##Label N 3
label_3 = Label(fenetre, text="Les données :", background="grey", font = ("arial", 10, "bold"))
label_3.place(x=20, y=300)

##Label N 3-1
label_3_1 = Label(fenetre, text="", justify = "left", background="grey", font = ("arial",11))
label_3_1.place(x=20, y=330)

##Label N 4-0 : Titre de l'appli
label_4_0 = Label(fenetre, text="Projet Libre : Détection de Fraude « SMS Alert – 'Spam Detector' »", background="grey", font = ("arial", 16, "bold"))
label_4_0.place(x=400, y=30)

##Label N 4 : Algos ML
label_4 = Label(fenetre, text="Choix de l'algorithme ML :", background="grey", font = ("arial", 10, "bold"))
label_4.place(x=440, y=100)

##Label N 5 : Valeurs clés
label_5 = Label(fenetre, text="Valeurs clés :", background="grey", font = ("arial", 10, "bold"))
label_5.place(x=440, y=300)

##Label N 6 : Accuracy (Précision, Robustesse)
label_6 = Label(fenetre, text="*Accuracy :", background="grey", font = ("arial", 12))
f6 = font.Font(label_6, label_6.cget("font"))
f6.configure(underline=True)
label_6.configure(font=f6)
label_6.place(x=460, y=330)

label_6_1 = Label(fenetre, text="", background="grey", font = ("arial", 12, "bold"))
label_6_1.place(x=550, y=330)

##Label N 7 : F1_Score (C'est la mesure de la précision du test)
label_7 = Label(fenetre, text="*F1_Score :", background="grey", font = ("arial", 12))
f7 = font.Font(label_7, label_7.cget("font"))
f7.configure(underline=True)
label_7.configure(font=f7)
label_7.place(x=460, y=360)

label_7_1 = Label(fenetre, text="", background="grey", font = ("arial", 12, "bold"))
label_7_1.place(x=550, y=360)

##Label N 13 : Temps d'éxecution (execution time)
label_13 = Label(fenetre, text="*Temps d'éxecution :", background="grey", font = ("arial", 12))
f13 = font.Font(label_13, label_13.cget("font"))
f13.configure(underline = True)
label_13.configure(font=f13)
label_13.place(x=460, y=390)

label_13_1 = Label(fenetre, text="",background="grey", font = ("arial", 12, "bold"))
label_13_1.place(x=620, y=390)

##Label N 8 : Confusion Matrix (C'est la Matrice de Confusion)
label_8 = Label(fenetre, text="*Confusion Matrix :", background="grey", font = ("arial", 12))
f8 = font.Font(label_8, label_8.cget("font"))
f8.configure(underline=True)
label_8.configure(font=f8)
label_8.place(x=460, y=420)

label_8_1 = Label(fenetre, text="", background="grey", font = ("arial", 12, "bold"))
label_8_1.place(x=620, y=420)

##Label N 9 : Notes (Accuracy + F1_Score + Confusion Matrix)
label_9 = Label(fenetre, text="** INFOS **", background="grey", font = ("arial", 10, "bold"))
label_9.place(x=440, y=470)

label_9_1 = Label(fenetre, text="", justify = "left" ,background="grey", font = ("arial", 10))
label_9_1.place(x=450, y=500)

label_9_1["text"] = '''#La matrice de confusion est une matrice qui mesure
la qualité d'un système de classification.
(*) Ligne L => classe réelle, Colonne C => classe éstimée.
(*) Cellule (ligne L, colonne C) => le nombre d'éléments de la classe
réelle L qui ont été estimés comme appartenant à la classe C.

#le score F1 (ou la mesure F ) est une mesure de la précision d'un test

#Accuracy (Précision, Robustesse) est la proportion d'items
pertinents parmi les items proposés.'''                                              


##Label N 10 : Visualisation (Heatmap)
label_10 = Label(fenetre, text="Visualisation (Heatmap):", background="grey", font = ("arial", 10, "bold"))
label_10.place(x=900, y=100)

##heatmap_img = Image.open("IMT_2.jpg")
##photo_Heatmap = ImageTk.PhotoImage(heatmap_img)
##photo_Heatmap_label = Label(fenetre, image = photo_Heatmap)
##photo_Heatmap_label.place(x=900, y=120)

##Label N 11 : Détails (Commenatires, Explications ...)
label_11 = Label(fenetre, text="Détails (Explications):", background="grey", font = ("arial", 10, "bold"))
label_11.place(x=900, y=470)    #910

label_11_1 = Label(fenetre, text="", justify = "left", background="grey", font = ("arial", 10))
label_11_1.place(x=900, y=500)

##Label N 12 : Copyright
label_12 = Label(fenetre, text="© Groupe N°5 (ADE 2020): Eddy, Paul, Badr", background="grey",  font = ("arial", 10,"bold"))
label_12.place(x=10, y=660)

filename = ""

def browse():
    global filename
    filename =  filedialog.askopenfilename(initialdir = "", title = "Séléctionner votre fichier de données", filetypes = (("TXT files","*.txt"),("All files","*.*"))) #Tuple de chemins
    #print(dir(filedialog))
    #print("filename", len(filename))
    if len(filename) != 0:
        #filename : Un seul fichier!
        print("Chemin du fichier:", type(filename))                    #type(filename) == <class 'str'>
        label_2["text"] = "Le fichier est chargé ... OK!"

        
    else :
        showerror("Erreur", "Veuillez inserer un seul ficher de données, SVP!")

    return print("Ici c'est 'browse'")

##Creation du boutton Browse
bouton_browse = Button(fenetre, text="Browse", font = ("arial", 10, "bold"), highlightbackground="grey", command=browse)
bouton_browse.place(x=327, y=180)



##Radio Button (Algorithms)
radioValue = IntVar()       #value = StringVar()
bouton1 = Radiobutton(fenetre, text="Naïve Bayes Algorithm", variable=radioValue,bg='grey', value=1)      #1 er choix ==> Native Bayes:
#bouton2 = Radiobutton(fenetre, text="Linear SVC Algorithm", variable=radioValue, value=2)        #2 eme choix ==> Linear SVC:
bouton3 = Radiobutton(fenetre, text="Random Forest Algorithm", variable=radioValue, bg='grey',value=3)     #3 eme choix ==> Random Forest:
bouton1.place(x=470, y=140)
#bouton2.place(x=470, y=170)
bouton3.place(x=470, y=200)


#X_train, X_test, y_train, y_test = list(), list(), list(), list()

def plot_heatmap(x):

    heatmap_img = Image.open(x)
    heatmap_img = heatmap_img.resize((430, 300), Image.ANTIALIAS)
    heatmap_img_tk = ImageTk.PhotoImage(heatmap_img)
    heatmap_img_label = Label(fenetre, image = heatmap_img_tk)
    heatmap_img_label.image = heatmap_img_tk
    heatmap_img_label.place(x=900, y=140)
    
    

def lancer():

    if radioValue.get() == 0:
        showerror("Erreur", "Ooops! Impossible de choisir l'algorithme.\n\n"+"Pensez à charger votre fichier de données, SVP!")
        return print("Attention: Pensez à charger votre fichier de données SVP")        #Interface d'information !!!

    else :     
        #print("les données :", df, "shape :", df.shape)
        #Afficher le fichier dans la zone corespondante!
        df = pd.read_csv(filename, sep='\t', header = None, names = ['y', 'message'])
        label_3_1["text"] = df
        print("df:", df)
        df.y = df.y.apply(lambda x: 1 if x == 'spam' else 0)  # dictionary = {ham:0, spam:1}

        # first split into train and test
        X_train, X_test, y_train, y_test = train_test_split(df.message, df.y, test_size=0.33, random_state=42)

        ## then vectorize (TF-IDF)
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X_train).toarray()
        X_test = vectorizer.transform(X_test).toarray()

        #print("Value", value)
        print("radioValue", radioValue.get())

        if radioValue.get() == 1:
            ## Naive Bayes classifier

            tic_NB = time.time()
            gnb = GaussianNB()
            y_gnb = gnb.fit(X_train, y_train).predict(X_test)
            tac_NB = time.time()

            time_NB = tac_NB - tic_NB

            print("Naive Bayes Confusion Matrix")
            CM_gnb = confusion_matrix(y_test, y_gnb)
            print(CM_gnb)
            print("Accuracy ", accuracy_score(y_test, y_gnb))
            print("F1 Score ", f1_score(y_test, y_gnb))
            print("Execution time ", round(time_NB,2),"s")

            label_6_1["text"] = str(round(accuracy_score(y_test, y_gnb)*100,2))+"%"     #Accuracy
            label_7_1["text"] = str(round(f1_score(y_test, y_gnb)*100,2))+"%"           #F1_Score   
            label_13_1["text"] = str(round(time_NB,2))+"s"                              #Time execution
            label_8_1["text"] = CM_gnb                                                  #Confusion Matrix 

            #Visualisation
            plot1 = plt.figure(1)
            print("Naive Bayes Visualization ...\n")
            sns.heatmap(CM_gnb, square=True, annot=True, fmt="d", cmap="RdBu", cbar=True, xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
            plt.title("Naive Bayes Heatmap\n")
            plt.xlabel("True Label")
            plt.ylabel("Predicted Label")
            #plt.show()
            plt.savefig('native_bayes.png')
            x = 'native_bayes.png'
            plot_heatmap(x)
            
            
            
            #Explications_Naive_Bayes
            label_11_1["text"] = '''Seulement 150 SMS sont des hams et classsifiés par le Naive Bayes
    comme étant des spams!
    (*) le niveau de prédiction et le F1_Score sont élevés!

    => On peut compter sur cet algorithme au développement
    de notre solution (√)'''
            
            return print("Ici c'est 'Native Bayes'")
            
    #     elif radioValue.get() == 2:
    #         ## SVM Classifier

    #         tic_LSVC = time.time()
    #         svc = SVC(gamma='auto')
    #         svc.fit(X_train, y_train)
    #         y_svc = svc.predict(X_test)
    #         tac_LSVC = time.time()

    #         time_LSVC = tac_LSVC - tic_LSVC
            
    #         print("SVC Confusion Matrix")
    #         CM_svc = confusion_matrix(y_test, y_svc)
    #         print(CM_svc)
    #         print("Accuracy ", accuracy_score(y_test, y_svc))
    #         print("F1 Score ", f1_score(y_test, y_svc))
    #         label_13_1["text"] = str(round(time_LSVC,2))+"s"

    #         label_6_1["text"] = str(round(accuracy_score(y_test, y_svc)*100,2))+"%"     #Accuracy
    #         label_7_1["text"] = str(round(f1_score(y_test, y_svc)*100,2))+"%"           #F1_Score   
    #         label_13_1["text"] = str(round(time_LSVC,2))+"s"                            #Time execution
    #         label_8_1["text"] = CM_svc                                                  #Confusion Matrix

    #         #Visualisation
    #         plot2 = plt.figure(2)
    #         print("SVC Visualization ...\n")
    #         sns.heatmap(CM_svc, square=True, annot=True, fmt="d", cmap="RdBu", cbar=True, xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    #         plt.title("SVC Heatmap\n")
    #         plt.xlabel("True Label")
    #         plt.ylabel("Predicted Label")
    #         #plt.show()
    #         plt.savefig('linear_svc.png')
    #         x = 'linear_svc.png'
    #         plot_heatmap(x)

    #         #Explications_SVC
    #         label_11_1["text"] = '''(*) L'algorithme Linear SVC n'a pas détectés de spams + F1_Score = 0

    # => Algorithme faible meme si le pourcentage de prédiction est important! (X)'''
            
    #         return print("Ici c'est 'Linear SVC'")
            
        elif radioValue.get() == 3:
            ## Random Forest CLassifier

            tic_RF = time.time()
            rf = RandomForestClassifier(max_depth=100, random_state=0)
            rf.fit(X_train, y_train)
            y_rf = rf.predict(X_test)
            tac_RF = time.time()

            time_RF = tac_RF - tic_RF

            print("Random Forest Confusion Matrix")
            CM_rf = confusion_matrix(y_test, y_rf)
            print(CM_rf)
            print("Accuracy ", accuracy_score(y_test, y_rf))
            print("F1 Score ", f1_score(y_test, y_rf))

            label_6_1["text"] = str(round(accuracy_score(y_test, y_rf)*100,2))+"%"      #Accuracy
            label_7_1["text"] = str(round(f1_score(y_test, y_rf)*100,2))+"%"            #F1_Score   
            label_13_1["text"] = str(round(time_RF,2))+"s"                             #Time execution
            label_8_1["text"] = CM_rf                                                   #Confusion Matrix
            
            #Visualisation
            plot3 = plt.figure(3)
            print("Random Forest Visualization ...\n")
            sns.heatmap(CM_rf, square=True, annot=True, fmt="d", cmap="RdBu", cbar=True, xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
            plt.title("Random Forest Heatmap\n")
            plt.xlabel("True Label")
            plt.ylabel("Predicted Label")
            #plt.show()
            plt.savefig('random_forest.png')
            x = 'random_forest.png'
            plot_heatmap(x)

            #Explications_Random_Forest
            label_11_1["text"] = '''(*) L'algorithme ne s'est pas trompé pour détécter de spams
    (*)Accuracy ~ 100% + F1_Score très élevé!

    => Algorithme Très fort et on peut l'adapter pour notre solution! (√)'''   
            
            return print("Ici c'est 'Random Forest'")
            
        else :
            showinfo("Erreur", "Attention: Veuillez choisir un algorithme SVP!")
            return print("Attention: Veuillez choisir un algorithme SVP!")          #Interface d'information !!!

##Creation du boutton Lancer
bouton_lancer = Button(fenetre, text="Lancer", highlightbackground="grey",font = ("arial", 10, "bold"), command=lancer)
bouton_lancer.place(x = 700, y = 170)

#Creation du boutton Quitter
bouton_quitter = Button(fenetre, text="Quitter", highlightbackground="grey", font = ("arial", 10, "bold"), command=quit)
bouton_quitter.place(x = 1275, y = 650)

# On démarre la boucle Tkinter qui s'interompt quand on ferme la fenêtre
fenetre.mainloop()
fenetre.destroy()