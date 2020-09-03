"""
Autor: Juan David Martínez Gualdrón
Asignatura: Aprendizaje Automático
Fecha: 23/06/2020
Maestría en Ingeniería Electrónica
Facultad de Ingeniería Eléctrica y Electrónica
Título del proyecto :  Modelo de aprendizaje automático en datos de la expresión genética
                       para la clasificación y predicción del cáncer de pecho
                       
"""

# --------------- IMPORTO LAS LIBRERÍAS A UTILIZAR ---------------------
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score, roc_auc_score
import scikitplot as skplt
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.linear_model import LassoCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf



# --------------- IMPORTO EL CONJUNTO DE DATOS ------------------------

data=pd.read_csv('Breast_GSE45827.csv')
print(data.head())

# --------------- ANÁLISIS DE LA DATA ---------------------------------

# Verifico si hay datos nulos o perdidos en el dataset
print('Datos nulos o perdidos en el conjunto de datos')
print(data.isnull().sum())

# Verifico los tipos de datos del dataset
print('Tipos de datos del dataset')
print(data.dtypes)

# Verifico la cantidad de datos o la forma del dataset
print('Cantidad de datos')
print(data.shape)

# Verifico la información de dataset
print('Información del conjunto de datos')
print(data.info())

# Descripción del dataset
print('Descripción del conjunto de datos')
description=data.describe()
print(description)

# ----------------- PREPROCESAMIENTO DE LOS DATOS ---------------------

# Elimino las columnas inncesarias para el analisis

# Se elimina la columna de muestras ya que no aporta información relevante
datac=data.drop(['samples'],axis=1)

# Cambio los datos del tipo de cáncer en datos numéricos
datac['type'].replace(['basal','HER','cell_line','normal','luminal_A','luminal_B'],
                      [1,2,3,4,5,6],inplace=True)

# Separo cada una de las clases
xa=np.array(datac[datac['type']==1]) # Clase basal
xb=np.array(datac[datac['type']==2]) # Clase HER
xc=np.array(datac[datac['type']==3]) # Clase cell_line
xd=np.array(datac[datac['type']==4]) # Clase normal
xe=np.array(datac[datac['type']==5]) # Clase luminal_A
xf=np.array(datac[datac['type']==6]) # Clase luminal_B

# Dado que las clases se encuentran desbalanceadas se hará un remuestreo
# Para que cada clase contenga 40 muestras 

# Clase basal
xa=xa[1:,:] # Elimino la utlima fila de la clase A para que tenga 40 muestras

# Clase HER
# Realizo un remuestreo para aumemtar el numero de muestras
bootb=resample(xb,replace=True,n_samples=10,random_state=1)
xbn=np.concatenate((xb,bootb),axis=0) # Concateno el nuevo vector de sobremuestras con clase b

# Clase cell_line
# Realizo un remuestreo para aumemtar el numero de muestras
bootc=resample(xc,replace=True,n_samples=26,random_state=1)
xcn=np.concatenate((xc,bootc),axis=0) # Concateno el nuevo vector de sobremuestras con clase c

# Clase normal
# Realizo un remuestreo para aumemtar el numero de muestras
bootd=resample(xd,replace=True,n_samples=33,random_state=1)
xdn=np.concatenate((xd,bootd),axis=0) # Concateno el nuevo vector de sobremuestras con clase d

# Clase luminal_A
# Realizo un remuestreo para aumemtar el numero de muestras
boote=resample(xe,replace=True,n_samples=11,random_state=1)
xen=np.concatenate((xe,boote),axis=0) # Concateno el nuevo vector de sobremuestras con clase e

# Clase luminal_B
# Realizo un remuestreo para aumemtar el numero de muestras
bootf=resample(xf,replace=True,n_samples=10,random_state=1)
xfn=np.concatenate((xf,bootf),axis=0) # Concateno el nuevo vector de sobremuestras con clase f

# Concateno todas las matrices de atributos
xab=np.concatenate((xa,xbn),axis=0)
xcd=np.concatenate((xcn,xdn),axis=0)
xef=np.concatenate((xen,xfn),axis=0)
xac=np.concatenate((xab,xcd),axis=0)
# Nuevo arreglo x_test que contiene los datos balanceados
datan=np.concatenate((xac,xef),axis=0)

# Separo los atributos de los labels
x=datan[:,1:]      # Atributos
y=datan[:,0]       # Labels o etiquetas

# Realizo un escalado a todos los datos de los atributos para que queden en la misma escala
scale=StandardScaler()  # Escalado 
x=scale.fit_transform(x)

# -------------------- VISUALIZACIÓN DE LOS DATOS ----------------------------

# Utilizo herramientas para visaualizar datos de alta dimensionalidad

#  -------------- ANÁLSIS DE COMPONENTES PRINCIPALES - PCA -------------------

# Dispersión de los datos normales con PCA
pca=PCA(n_components=2) # Número de componentes a transformar
data_transform=pca.fit_transform(datac)

# Grafico el nuevo conjunto de datos con PCA
figurepca=mglearn.discrete_scatter(data_transform[:,0], data_transform[:,1],data['type'])
plt.legend(loc='center left' , bbox_to_anchor=(1, 0.5))
plt.title('Dispersión de los datos con PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid()
plt.show(figurepca)

#  -------------- INCRUSTACIÓN DE VECINOS ESTOCÁSTICOS - TSNE -------------------

# Dispersión de los datos normales con TSNE
tsne=TSNE(n_components=2) # Número de componentes a transformar
data_tsne=tsne.fit_transform(datac)

# Grafico el nuevo conjunto de datos con TSNE
figuretsne=mglearn.discrete_scatter(data_tsne[:,0], data_tsne[:,1],data['type'])
plt.legend(loc='center left' , bbox_to_anchor=(1, 0.5))
plt.title('Dispersión de los datos con TSNE')
plt.xlabel('TSNE 1')
plt.ylabel('TSNE 2')
plt.grid()
plt.show(figuretsne)

# --------- DIVISIÓN DEL CONJUNTO DE DATOS EN ENTRENAMIENTO Y PRUEBA -------------

# Divido los datos en entrenamiento 70 % y prueba 30 %
# Me aseguro que cada clase tenga el mismo número de datos tanto para train como para test
x_traina , x_testa , y_traina , y_testa = train_test_split(xa,y[y==1],test_size=0.3,random_state=1)
x_trainb , x_testb , y_trainb , y_testb = train_test_split(xbn,y[y==2],test_size=0.3,random_state=1)
x_trainc , x_testc , y_trainc , y_testc = train_test_split(xcn,y[y==3],test_size=0.3,random_state=1)
x_traind , x_testd , y_traind , y_testd = train_test_split(xdn,y[y==4],test_size=0.3,random_state=1)
x_traine , x_teste , y_traine , y_teste = train_test_split(xen,y[y==5],test_size=0.3,random_state=1)
x_trainf , x_testf , y_trainf , y_testf = train_test_split(xfn,y[y==6],test_size=0.3,random_state=1)

# Concateno todas las matrices de atributos de train
xabt=np.concatenate((x_traina,x_trainb),axis=0)
xcdt=np.concatenate((x_trainc,x_traind),axis=0)
xeft=np.concatenate((x_traine,x_trainf),axis=0)
xact=np.concatenate((xabt,xcdt),axis=0)
# Nuevo arreglo x_train que contiene los datos balanceados
x_train=np.concatenate((xact,xeft),axis=0)

# Concateno todas las matrices de atributos de test
xabp=np.concatenate((x_testa,x_testb),axis=0)
xcdp=np.concatenate((x_testc,x_testd),axis=0)
xefp=np.concatenate((x_teste,x_testf),axis=0)
xacp=np.concatenate((xabp,xcdp),axis=0)
# Nuevo arreglo x_test que contiene los datos balanceados
x_test=np.concatenate((xacp,xefp),axis=0)

# Concateno todas las matrices de labels de train
yabt=np.concatenate((y_traina,y_trainb),axis=0)
ycdt=np.concatenate((y_trainc,y_traind),axis=0)
yeft=np.concatenate((y_traine,y_trainf),axis=0)
yact=np.concatenate((yabt,ycdt),axis=0)
# Nuevo arreglo y_train que contiene los datos balanceados
y_train=np.concatenate((yact,yeft),axis=0)

# Concateno todas las matrices de labels de test
yabp=np.concatenate((y_testa,y_testb),axis=0)
ycdp=np.concatenate((y_testc,y_testd),axis=0)
yefp=np.concatenate((y_teste,y_testf),axis=0)
yacp=np.concatenate((yabp,ycdp),axis=0)
# Nuevo arreglo y_test que contiene los datos balanceados
y_test=np.concatenate((yacp,yefp),axis=0)


# Visualización de la división de los datos con PCA
pca=PCA(n_components=2) # Número de componentes a transformar
x_train_transform=pca.fit_transform(x_train)
x_test_transform=pca.fit_transform(x_test)
plt.scatter(x_train_transform[:,0], x_train_transform[:,1],c='red',label='Training')
plt.scatter(x_test_transform[:,0], x_test_transform[:,1],c='blue',label='Testing')
plt.legend(loc='center left' , bbox_to_anchor=(1, 0.5))
plt.title('División de los datos en entrenamiento y prueba')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid()

# ------------------- IMPLEMENTACIÓN DE LOS MODELOS DE ML --------------------------
x_test=np.delete(x_test,0,1)
x_train=np.delete(x_train,0,1)

# Realizo un cross validation para cada modelo

# SVM - RBF
print('Evaluación del modelo SVM-RBF')
svm=SVC(kernel='rbf',C=1.4)
scoresSVM=cross_val_score(svm,x_train,y_train,cv=10)
print('Accuracy SVM : %0.2f (+/- %0.2f)' % (scoresSVM.mean(), scoresSVM.std() * 2))


# SVM - Lineal
print('Evaluación del modelo SVM-Lineal')
svml=SVC(kernel='linear',C=1.1,tol=0.5e-3)
scoresSVML=cross_val_score(svml,x_train,y_train,cv=10)
print('Accuracy SVML : %0.2f (+/- %0.2f)' % (scoresSVML.mean(), scoresSVML.std() * 2))

# KNN
print('Evaluación del modelo KNN')
knn=KNeighborsClassifier(n_neighbors=9,metric='chebyshev')
scoresKNN=cross_val_score(knn,x_train,y_train,cv=10)
print('Accuracy KNN : %0.2f (+/- %0.2f)' % (scoresKNN.mean(), scoresKNN.std() * 2))

# Decision Trees
print('Evaluación del modelo DT')
dt=DecisionTreeClassifier(criterion='entropy',max_depth=8)
scoresDT=cross_val_score(dt,x_train,y_train,cv=10)
dt.fit(x_train,y_train)
print('Accuracy DT : %0.2f (+/- %0.2f)' % (scoresDT.mean(), scoresDT.std() * 2))

# Random Forest
print('Evaluación del modelo RF')
rf=RandomForestClassifier(n_estimators=20,max_depth=2)
scoresRF=cross_val_score(rf,x_train,y_train,cv=10)
rf.fit(x_train,y_train)
print('Accuracy RF : %0.2f (+/- %0.2f)' % (scoresRF.mean(), scoresRF.std() * 2))

print('Evaluación del modelo MLP')
mlp=MLPClassifier(hidden_layer_sizes=(100,100,100,100),activation='relu',max_iter=300,learning_rate_init=0.03)
scoresMLP=cross_val_score(mlp,x_train,y_train,cv=10)
mlp.fit(x_train,y_train)
print('Accuracy MLP : %0.2f (+/- %0.2f)' % (scoresMLP.mean(), scoresMLP.std() * 2))

# ------------------------------ FEATURE SELECTION --------------------------------------

# Realizo una selección de características para determinar cuales atributos son los 
# más importantes para el análisis

# Feature Selection using SelectFromModel
print('Shape de los atributos anterior : ', x.shape)
m=SelectFromModel(LassoCV())
m.fit(x,y)
nuevax=m.transform(x)
print('Shape de los nuevos atributos : ', nuevax.shape)

# Divido el nuevo conjunto de atributos cada una de las clases
datanueva=np.column_stack((y, nuevax))
xanew=np.array(datanueva[datanueva[:,0]==1]) # Clase basal
xbnew=np.array(datanueva[datanueva[:,0]==2]) # Clase HER
xcnew=np.array(datanueva[datanueva[:,0]==3]) # Clase cell_line
xdnew=np.array(datanueva[datanueva[:,0]==4]) # Clase normal
xenew=np.array(datanueva[datanueva[:,0]==5]) # Clase luminal_A
xfnew=np.array(datanueva[datanueva[:,0]==6]) # Clase luminal_B

# Separo en atributos y labels
xnew=datanueva[:,1:]      # Atributos
ynew=datanueva[:,0] 

# Divido los datos 70 % y 30 %
x_trainan , x_testan , y_trainan , y_testan = train_test_split(xanew,ynew[y==1],test_size=0.3,random_state=1)
x_trainbn , x_testbn , y_trainbn , y_testbn = train_test_split(xbnew,ynew[y==2],test_size=0.3,random_state=1)
x_traincn , x_testcn , y_traincn , y_testcn = train_test_split(xcnew,ynew[y==3],test_size=0.3,random_state=1)
x_traindn , x_testdn , y_traindn , y_testdn = train_test_split(xdnew,ynew[y==4],test_size=0.3,random_state=1)
x_trainen , x_testen , y_trainen , y_testen = train_test_split(xenew,ynew[y==5],test_size=0.3,random_state=1)
x_trainfn , x_testfn , y_trainfn , y_testfn = train_test_split(xfnew,ynew[y==6],test_size=0.3,random_state=1)

# Concateno todas las matrices de atributos de train
xabtn=np.concatenate((x_trainan,x_trainbn),axis=0)
xcdtn=np.concatenate((x_traincn,x_traindn),axis=0)
xeftn=np.concatenate((x_trainen,x_trainfn),axis=0)
xactn=np.concatenate((xabtn,xcdtn),axis=0)
# Nuevo arreglo x_train que contiene los datos balanceados
x_trainnew=np.concatenate((xactn,xeftn),axis=0)

# Concateno todas las matrices de atributos de test
xabpn=np.concatenate((x_testan,x_testbn),axis=0)
xcdpn=np.concatenate((x_testcn,x_testdn),axis=0)
xefpn=np.concatenate((x_testen,x_testfn),axis=0)
xacpn=np.concatenate((xabpn,xcdpn),axis=0)
# Nuevo arreglo x_test que contiene los datos balanceados
x_testnew=np.concatenate((xacpn,xefpn),axis=0)

# Concateno todas las matrices de labels de train
yabtn=np.concatenate((y_trainan,y_trainbn),axis=0)
ycdtn=np.concatenate((y_traincn,y_traindn),axis=0)
yeftn=np.concatenate((y_trainen,y_trainfn),axis=0)
yactn=np.concatenate((yabtn,ycdtn),axis=0)
# Nuevo arreglo y_train que contiene los datos balanceados
y_trainn=np.concatenate((yactn,yeftn),axis=0)

# Concateno todas las matrices de labels de test
yabpn=np.concatenate((y_testan,y_testbn),axis=0)
ycdpn=np.concatenate((y_testcn,y_testdn),axis=0)
yefpn=np.concatenate((y_testen,y_testfn),axis=0)
yacpn=np.concatenate((yabpn,ycdpn),axis=0)
# Nuevo arreglo y_test que contiene los datos balanceados
y_testnn=np.concatenate((yacpn,yefpn),axis=0)

# Elimino la columna de label en los atributos
x_testnn=np.delete(x_testnew,0,1)
x_trainn=np.delete(x_trainnew,0,1)


# Implemento los modelos con el conjunto de datos seleccionado

# Realizo un cross validation para cada modelo

# SVM - RBF - FS
print('Evaluación del modelo SVM-RBF con Feature Selection')
svmfs=SVC(kernel='rbf',C=1)
scoresSVMfs=cross_val_score(svmfs,x_trainn,y_trainn,cv=10)
svmfs.fit(x_trainn,y_trainn)
print('Accuracy SVM : %0.2f (+/- %0.2f)' % (scoresSVMfs.mean(), scoresSVMfs.std() * 2))

# SVM - Lineal - FS
print('Evaluación del modelo SVM-Lineal con Feature Selection')
svmlfs=SVC(kernel='linear',C=1.1,tol=0.5e-3)
scoresSVMLfs=cross_val_score(svmlfs,x_trainn,y_trainn,cv=10)
svmlfs.fit(x_trainn,y_trainn)
print('Accuracy SVML : %0.2f (+/- %0.2f)' % (scoresSVMLfs.mean(), scoresSVMLfs.std() * 2))

# KNN - FS
print('Evaluación del modelo KNN con Feature Selection')
knnfs=KNeighborsClassifier(n_neighbors=7,metric='chebyshev')
scoresKNNfs=cross_val_score(knnfs,x_trainn,y_trainn,cv=10)
print('Accuracy KNN : %0.2f (+/- %0.2f)' % (scoresKNNfs.mean(), scoresKNNfs.std() * 2))

# Decision Trees - FS
print('Evaluación del modelo DT con Feature Selection')
dtfs=DecisionTreeClassifier(criterion='entropy',max_depth=2)
scoresDTfs=cross_val_score(dtfs,x_trainn,y_trainn,cv=10)
print('Accuracy DT : %0.2f (+/- %0.2f)' % (scoresDTfs.mean(), scoresDTfs.std() * 2))

# Random Forest - FS
print('Evaluación del modelo RF con Feature Selection')
rffs=RandomForestClassifier(n_estimators=20,max_depth=8)
scoresRFfs=cross_val_score(rffs,x_trainn,y_trainn,cv=10)
print('Accuracy RF : %0.2f (+/- %0.2f)' % (scoresRFfs.mean(), scoresRFfs.std() * 2))


print('Evaluación del modelo MLP')
mlp=MLPClassifier(hidden_layer_sizes=(100,100,100,100),activation='tanh',learning_rate_init=0.05)
scoresMLP=cross_val_score(mlp,x_trainn,y_trainn,cv=10)
mlp.fit(x_trainn,y_trainn)
print('Accuracy MLP : %0.2f (+/- %0.2f)' % (scoresMLP.mean(), scoresMLP.std() * 2))


# ---------------------------- PRUEBA DE LOS MODELOS CON DATOS DE PRUEBA -----------------------------

class_names = ('basal','HER','cell_line','normal','luminal_A','luminal_B')

#---------------- Predicción del Random Forest ------------------------------

classifier=SVC(kernel='rbf', probability=True).fit(x_train,y_train)


# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, x_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
y_pred_rbf=classifier.predict(x_test)
print('Accuracy RBF : ', accuracy_score(y_test,y_pred_rbf))

# Grafico la Curva ROC - AUC de SVM RBF
y_proba=classifier.predict_proba(x_test)
skplt.metrics.plot_roc_curve(y_test, y_proba)
plt.show()

print('F1 Score SVM RBF : ', f1_score(y_test,y_pred_rbf,average='micro'))

#---------------- Predicción del MLP ------------------------------

# Predicción del MLP
classifier2=MLPClassifier(hidden_layer_sizes=(200,100),activation='tanh',learning_rate_init=0.05).fit(x_trainn,y_trainn)


# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier2, x_testnn, y_testnn,
                                 display_labels=class_names,
                                 cmap=plt.cm.Reds,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
y_pred_mlp=classifier2.predict(x_testnn)
print('Accuracy MLP : ', accuracy_score(y_testnn,y_pred_mlp))

# Grafico la Curva ROC - AUC de MLP
y_proba_mlp=mlp.predict_proba(x_testnn)
skplt.metrics.plot_roc_curve(y_testnn, y_proba_mlp)
plt.show()

print('F1 Score MLP : ', f1_score(y_testnn,y_pred_mlp,average='micro'))




