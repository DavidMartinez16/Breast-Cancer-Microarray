# Breast Cancer Microarray Predictor - Project Overview

_This is a Machine Learning Project which uses a dataset that contains a microarray with more than 54000 gene expressions of breast cancer. The target was to develop an algorithm capable of analyze and predict different breast cancer types, such as luminal A, luminal B, HER, Basal, Cell Line and Normal Tissue._

## General Information
* [Dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45827) - The dataset was downloaded in the Curated Microarray Database Repository, which contains several databases about different kinds of cancer.
* Created a machine learning models that classify the type of breast cancer by analyzing 54675 gene expressions. Then, using feature selection, tested the performance of the previous machine learning models but using only 153 gene expressions, resulting in better evaluation models metrics and a remarkable reduction in models training time.
* The model with the best performance using **54675genes**  was the **Random Forest Classifier**. On the other hand, the models with higher metrics using **153 genes** were the **Multilayer Perceptron and Support Vector Machines with RBF Kernel**.
* The evaluation metrics were F1-Score, ROC Curve and Accuracy. 
* The models used in this project were:
  * Random Forest
  * Support Vector Machine ( Lineal and RBF )
  * K-Nearest Neighbors
  * Multilayer Perceptron
  * Decision Trees

## Resources
* **Python Version:** 3.7
* **Packages:** Pandas, Numpy, Matplotlib, Seaborn, Sklearn, Mglearn and Scikitplot
* **Programs:** Anaconda, Spyder and Jupyter Notebook

# Repository Description

_The repository contains two folders named **Unnbalanced Data** and **Balanced Data**. In the first folder you will find the Data Cleaning, the Exploratory Data Analysis and the Model Implementation with the unnbalanced classes. And in the second folder you will find a Python File which contains the same Data Cleaning, the balancing of the classes using Bootstrapping and the models evaluations in the same Python file._

## Exploratory Data Analysis

For this case I used dimensionality reduction techniques in order to graph the dispersion of the 6 classed in 2 dimensions, such as **PCA** and **TSNE**.

## PCA
![pca](https://user-images.githubusercontent.com/63115543/92185644-12c9d900-ee1a-11ea-9920-1c368a2a288d.jpg)

## TSNE
![TSNE](https://user-images.githubusercontent.com/63115543/92185652-16f5f680-ee1a-11ea-8fc5-ff6d0775151b.jpg)

# Models Performance

## 54675 Genes

### Random Forest

In the figure you can see the confussion matrix of this model, and the obtained **F1 Score** was **86.95 %**. In this case i used the F1 Score to evaluate the performance of the model due to unnbalanced classes.

![rf](https://user-images.githubusercontent.com/63115543/92185792-69371780-ee1a-11ea-99a5-14f7c85e1607.jpg)

## 153 Genes

For this case, I used the Support Vector Machines with Kernel RBF and the Multilayer Perceptron. The model with less error was the SVM, however, both models predict with a high accuracy value the different types of cancer. It's important to mention that in this case the classes were balanced using the Bootstrapping technique.

### SVM (RBF) 

**Metrics:**
  * **Accuracy:** 98.61 %
  
![svm_cm](https://user-images.githubusercontent.com/63115543/92185996-fd08e380-ee1a-11ea-8828-903fff01d4fe.png)

### MLP 

**Metrics:**
  * **Accuracy:** 97.22 %

![mlp_cm](https://user-images.githubusercontent.com/63115543/92186007-085c0f00-ee1b-11ea-8ec4-f40fa2d7e73f.png)
