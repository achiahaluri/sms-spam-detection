from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import tkinter as ttk
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import csv
import sklearn
#import pickle
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV,train_test_split,StratifiedKFold,cross_val_score,learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,f1_score
from tkinter.simpledialog import askstring
import string
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

main = tkinter.Tk()
main.title("SMS Spam Detection")
main.geometry("1300x1200")

global dataset_file
global train_features, label_features
global machine_learning
global data,text
global X_train, X_test, y_train, y_test
X_train = None
y_train = None
X_test = None
y_test = None
vectorizer = None
mnb = None
data = None


def loadData():
    global dataset_file
    global data,shape_label
    dataset_file = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=dataset_file)
    text.delete('1.0', END)
    text.insert(END,dataset_file+" dataset loaded\n\n")
    data = pd.read_csv(dataset_file, encoding='latin-1')
    text.insert(END,str(data.head())+"\n")
    # Calculate and display the shape of the data
    dataset_shape = data.shape
    text.insert(tk.END, f"Dataset Shape: {dataset_shape}\n")
    # Identify missing values and display them
    missing_values = data.isnull().sum()
    # Display missing values in the text widget
    text.insert(tk.END, "Missing Values:\n")
    for column, count in missing_values.items():
        text.insert(tk.END, f"{column}: {count}\n")
def null():
    global data, text
    text.delete('1.0', END)
    # Check for missing values
    missing_values = data.isnull().sum()
    # Display missing values in the text widget
    text.insert(tk.END, "Missing Values:\n")
    for column, count in missing_values.items():
        text.insert(tk.END, f"{column}: {count}\n")
    # Visualize missing values using Matplotlib
    plt.figure(figsize=(8, 6))
    plt.bar(missing_values.index, missing_values.values)
    plt.xlabel('Columns')
    plt.ylabel('Missing Values Count')
    plt.title('Missing Values Visualization')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    columns_to_drop = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
    data = pd.DataFrame(data).drop(columns=columns_to_drop)
    missing_values = data.isnull().sum()
    # Display missing values in the text widget
    text.insert(tk.END, "Missing Values:\n")
    for column, count in missing_values.items():
        text.insert(tk.END, f"{column}: {count}\n")
    # Visualize missing values using Matplotlib
    plt.figure(figsize=(8, 6))
    plt.bar(missing_values.index, missing_values.values)
    plt.xlabel('Columns')
    plt.ylabel('Missing Values Count')
    plt.title('Missing Values Visualization')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    dataset_shape = data.shape
    text.insert(tk.END, f"Dataset Shape: {dataset_shape}\n")
    text.insert(END,str(data.head())+"\n")

    
def labelen():
    global data, text
    text.delete('1.0', END)
    data = data.rename(columns={"v2" : "Text", "v1":"Class"})
    text.insert(END,"After rename 'v1':'Class','v2':'Text'"+"\n")
    text.insert(END,str(data.head())+"\n")
    text.insert(END,"After map 'ham': 0, 'spam': 1"+"\n")
    data['numClass'] = data['Class'].map({'ham': 0, 'spam': 1})
    text.insert(END,str(data.head())+"\n")
     # Get the value counts of the 'label' column
    label_counts = data['Class'].value_counts()
    # Display the value counts in the text widget
    text.insert(END, "Value Counts of 'label' column:\n")
    for Class, count in label_counts.items():
        text.insert(END, f"{Class}: {count}\n")
    # Step 1: Identify and Display Null Values
    missing_values = data.isnull().sum()

    # Display missing values in the 'Class' and 'Text' columns
    if missing_values['Class'] > 0:
        text.insert(tk.END, "Missing Values in 'Class' column: {}\n".format(missing_values['Class']))

    if missing_values['Text'] > 0:
        text.insert(tk.END, "Missing Values in 'Text' column: {}\n".format(missing_values['Text']))

    # Step 2: Handle Null Values (for example, fill with the mode for 'Class' column)
    data['Class'].fillna(data['Class'].mode()[0], inplace=True)

    # Display that null values have been handled
    text.insert(tk.END, "Null values in 'Class' column have been filled.\n")

    # Step 3: Handle Null Values (for example, fill with the mode for 'Text' column)
    data['Class'].fillna(data['Text'].mode()[0], inplace=True)

    # Display that null values have been handled
    text.insert(tk.END, "Null values in 'Text' column have been filled.\n")
   
    
def word():
    global data, text
    text.delete('1.0', END)
    ham_words = ''
    spam_words = ''
    # Creating a corpus of spam messages
    for val in data[data['Class'] == 'spam'].Text:
        text_lower = val.lower()
        tokens = nltk.word_tokenize(text_lower)
        for words in tokens:
            spam_words = spam_words + words + ' '

    # Creating a  corpus of ham messages
    for val in data[data['Class'] == 'ham'].Text:
        text_lower = val.lower()
        tokens = nltk.word_tokenize(text_lower)
        for words in tokens:
            ham_words = ham_words + words + ' '

    spam_wordcloud = WordCloud(width=500, height=300).generate(spam_words)
    #Spam Word cloud
    plt.figure( figsize=(10,8), facecolor='w')
    plt.imshow(spam_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    ham_wordcloud = WordCloud(width=500, height=300).generate(ham_words)
    #Creating Ham wordcloud
    plt.figure( figsize=(10,8), facecolor='g')
    plt.imshow(ham_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

def corpus():
    global data, text, X_train,X_test, y_test, y_train,text_widget,vectorizer

    font1 = ('times', 12, 'bold')
    text_widget = Text(main, height=20, width=150)  # Create a Text widget for displaying text
    text_widget.place(x=10, y=250)
    text_widget.config(font=font1)
    text_widget.delete('1.0', END)

    def text_process(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
        return " ".join(text)
    
    data['Text'] = data['Text'].apply(text_process)
    text1 = pd.DataFrame(data['Text'])
    label = pd.DataFrame(data['Class'])

    # Counting how many times a word appears in the dataset
    total_counts = Counter()
    for i in range(len(text1)):
        for word in text1.values[i][0].split(" "):
            total_counts[word] += 1

    text_widget.insert(tk.END, "Total words in data set: " + str(len(total_counts)) + "\n")
    print("Total words in data set: ", len(total_counts))

    # Sorting in decreasing order (Word with highest frequency appears first)
    vocab = sorted(total_counts, key=total_counts.get, reverse=True)
    print(vocab[:60])
    text_widget.insert(tk.END, "Sorting in decreasing order (Word with highest frequency appears first) " + ' '.join(vocab[:60]) + "\n")

    # Mapping from words to index
    vocab_size = len(vocab)
    word2idx = {}

    #print vocab_size
    for i, word in enumerate(vocab):
        word2idx[word] = i

    # Text to Vector
    def text_to_vector(text1):
        word_vector = np.zeros(vocab_size)
        for word in text1.split(" "):
            if word2idx.get(word) is None:
                continue
            else:
                word_vector[word2idx.get(word)] += 1
        return np.array(word_vector)
    
    # Convert all titles to vectors
    word_vectors = np.zeros((len(text1), len(vocab)), dtype=np.int_)
    for i, (_, text1_) in enumerate(text1.iterrows()):
        word_vectors[i] = text_to_vector(text1_[0])
     

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data['Text'])
    
    text_widget.insert(tk.END, f"Word Vectors Shape: {word_vectors.shape}\n")
    text_widget.insert(tk.END, f"TF-IDF Vectors Shape: {vectors.shape}\n")

    #features = word_vectors
    features = vectors

    text_widget.insert(tk.END, f"features: {features}\n")   

    global X_train, X_test, y_train, y_test

    #split the dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split(features, data['Class'], test_size=0.15, random_state=111)

    # Show the results of the split in the Text widget
    text_widget.insert(END, "Training set has {} samples.\n".format(X_train.shape[0]))
    text_widget.insert(END, "Testing set has {} samples.\n".format(X_test.shape[0]))

    print(X_train.shape[0])
    print(X_test.shape[0])
    #y_train = y_train.values.reshape(-1, 1)  # Reshape to a 2D array
    #y_test = y_test.values.reshape(-1, 1)    # Reshape to a 2D array    

def model():
    global data, text, X_train,X_test, y_train, y_test, text_widget,mnb

    text_widget = Text(main, height=20, width=150)
    text_widget.place(x=10, y=250)
    text_widget.config(font=font1)
    text_widget.delete('1.0', END)

    #corpus()

    #initialize multiple classification models 
    svc = SVC(kernel='sigmoid', gamma=1.0)
    knc = KNeighborsClassifier(n_neighbors=49)
    mnb = MultinomialNB(alpha=0.2)
    dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
    lrc = LogisticRegression(solver='liblinear', penalty='l1')
    rfc = RandomForestClassifier(n_estimators=31, random_state=111)

    #create a dictionary of variables and models
    clfs = {'SVC' : svc,'KN' : knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 'RF': rfc}

    #fit the data onto the models
    def train(clf, features, targets):    
        clf.fit(features, targets)
        #predict(clf, features)

    def predict(clf, features):
        return (clf.predict(features))
    
    pred_scores_word_vectors = []
    for k,v in clfs.items():
        train(v, X_train, y_train)
        pred = predict(v, X_test)
        pred_scores_word_vectors.append((k, [accuracy_score(y_test , pred)]))
    print(pred_scores_word_vectors)
    #text_widget.insert(END, "pred_scores_word_vectors: {}\n".format(pred_scores_word_vectors)+"\n")
    text_widget.insert(END, "Model Scores:\n")
    for model_name, scores in pred_scores_word_vectors:
        text_widget.insert(END, f"{model_name}: Accuracy - {scores[0]:.4f}\n")

   
        
    
    





def prediction():
    global data, text, text_widget, vectorizer, mnb,x
    text_widget = Text(main, height=20, width=150)
    text_widget.place(x=10, y=250)
    text_widget.config(font=font1)
    text_widget.delete('1.0', END)

    # Get the user input from the Entry widget
    user_input = askstring("User Input", "Enter the text:")

    # Check if the user input is empty or canceled
    if user_input is None or user_input.strip() == "":
        text_widget.insert(tk.END, "No input provided.\n")
        return
    

    newtext = [user_input]
    integers = vectorizer.transform(newtext)

    x = mnb.predict(integers)

    if x == 1:
        print ("Message is SPAM")
    else:
        print ("Message is NOT Spam")
    text_widget.insert(tk.END, str(x) + "\n")

def matrix():
    global data, text, text_widget,X_train,X_test, y_train, y_test,vectorizer, mnb,x

    text_widget = Text(main, height=20, width=150)
    text_widget.place(x=10, y=250)
    text_widget.config(font=font1)
    text_widget.delete('1.0', END)

    y_pred_nb = mnb.predict(X_test)
    y_true_nb = y_test
    cm = confusion_matrix(y_true_nb, y_pred_nb)
    f, ax = plt.subplots(figsize =(5,5))
    sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
    plt.xlabel("y_pred_nb")
    plt.ylabel("y_true_nb")
    plt.show()
    




font = ('times', 16, 'bold')
title = Label(main, text='SMS Spam Detection')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload  Dataset", command=loadData)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

font1 = ('times', 13, 'bold')
nullButton = Button(main, text="Handel Null Values", command=null)
nullButton.place(x=50,y=150)
nullButton.config(font=font1)

font1 = ('times', 13, 'bold')
enButton = Button(main, text="Leable Encoding", command=labelen)
enButton.place(x=280,y=150)
enButton.config(font=font1)


font1 = ('times', 13, 'bold')
wordButton = Button(main, text="Word to Vector", command=word)
wordButton.place(x=480,y=150)
wordButton.config(font=font1)

font1 = ('times', 13, 'bold')
corpButton = Button(main, text="Corpus Building", command=corpus)
corpButton.place(x=680,y=150)
corpButton.config(font=font1)

font1 = ('times', 13, 'bold')
modelButton = Button(main, text="Model_Building", command=model)
modelButton.place(x=50,y=200)
modelButton.config(font=font1)

font1 = ('times', 13, 'bold')
predButton = Button(main, text="Model_Predection", command=prediction)
predButton.place(x=280,y=200)
predButton.config(font=font1)

font1 = ('times', 13, 'bold')
matrixButton = Button(main, text="Classification_Results", command=matrix)
matrixButton.place(x=480,y=200)
matrixButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)

main.config(bg='green')
main.mainloop()
