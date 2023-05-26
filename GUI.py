import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import preprocessor as p
import plotly.express as px
import tweepy
import csv
import pandas as pd
import numpy as np
from plotly import graph_objs as go
import datetime
import seaborn as sns
import nltk
import re
import string
from textblob import TextBlob
import matplotlib.pyplot as plt
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Download Required NLTK Files.
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
#%%  main window

window = tk.Tk()
window.geometry("1500x800") 
window.wm_title("Eren Bozkurt - CCT_CA2_2022-23")


frame1= tk.Frame(window, width =750, height = 800) 
frame1.grid(row=0, column=0)
frame2= tk.Frame(window, width =750, height = 800 ) 
frame2.grid(row=0, column=1)

var1 = tk.StringVar()
var2 = tk.StringVar()
var3 = tk.StringVar()
var4 = tk.StringVar()

label1 = tk.Label( frame1, textvariable=var1,font=("Arial", 15))
label2 = tk.Label( frame1, textvariable=var2, font=("Arial", 11))
label3 = tk.Label( frame2, textvariable=var3,font=("Arial", 15))
label4 = tk.Label( frame2, textvariable=var4, font=("Arial", 11))

twitterDataBefore, twitterDataAfter = "", ""
dataBefore , dataAfter = "",""
canvas = ''
dataModelling=""
X_train, X_test, y_train, y_test, X_test_transformed ="", "","","", ""
 

#%%
def OpenData():
    global twitterDataBefore, twitterDataAfter , dataBefore , dataAfter
    
    data1 = pd.read_csv("eurovision1.csv")
    data2 = pd.read_csv("eurovision2.csv")
    data3 = pd.read_csv("eurovision3.csv")
    data4 = pd.read_csv("eurovision4.csv")
    data5 = pd.read_csv("eurovision5.csv")
    data6 = pd.read_csv("eurovision6.csv")
    data7 = pd.read_csv("eurovision7.csv")
    data8 = pd.read_csv("eurovision8.csv")
    
    twitterDataBefore = pd.concat([data1, data2, data3, data4], ignore_index=True)
    goster1 = twitterDataBefore.loc[:,["text", "created_at"]]
    
    twitterDataAfter = pd.concat([data5, data6, data7, data8], ignore_index=True) 
    goster2= twitterDataAfter.loc[:,["text", "created_at"]]
    
    dataBefore = twitterDataBefore
    dataAfter = twitterDataAfter
    
    var1.set("Before the Eurovision Final")
    label1.place(relx =0.01, rely = 0.01)
    var2.set(goster1.head(30))
    label2.place(relx =0.01, rely = 0.09)   
    
    var3.set("After the Eurovision Final")
    label3.place(relx =0.01, rely = 0.01)
    var4.set(goster2.head(30))
    label4.place(relx =0.01, rely = 0.09)  
    
#%%

def DataFirstCleaning():
    global twitterDataBefore, twitterDataAfter , dataBefore , dataAfter

    # BEFORE 
    def preprocess_tweet(row):
        import preprocessor as p
        text = row['text']
        text = p.clean(text)
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text
    
    dataBefore['clean_tweet'] = dataBefore.apply(preprocess_tweet, axis=1)
    
    stop = stopwords.words('english')
    dataBefore['clean_tweet_stopwords'] = dataBefore['clean_tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    # Data After
    def preprocess_tweet(row):
        import preprocessor as p
        text = row['text']
        text = p.clean(text)
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text
    dataAfter['clean_tweet'] = dataAfter.apply(preprocess_tweet, axis=1)

    stop = stopwords.words('english')
    dataAfter['clean_tweet_stopwords'] = dataAfter['clean_tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    goster1 = dataBefore.loc[:,["text", "clean_tweet"]]
    goster2 = dataAfter.loc[:,["text", "clean_tweet"]]

    var1.set("Before the Eurovision Final Clean Data")
    label1.place(relx =0.01, rely = 0.01)
    var2.set(goster1.head(30))
    label2.place(relx =0.01, rely = 0.09)   
    
    var3.set("After the Eurovision Final Clean Data")
    label3.place(relx =0.01, rely = 0.01)
    var4.set(goster2.head(30))
    label4.place(relx =0.01, rely = 0.09) 

#%%

def usedWords():
    global twitterDataBefore, twitterDataAfter , dataBefore , dataAfter
   
    a = dataBefore['clean_tweet_stopwords'].str.cat(sep=' ')
    words = nltk.tokenize.word_tokenize(a)
    word_dist = nltk.FreqDist(words)
    dff1 = pd.DataFrame(word_dist.most_common(), 
                        columns=['Word', 'Frequency'])
    dff1['Word_Count'] = dff1.Word.apply(len)
    dff1[:10]
    
    a = dataAfter['clean_tweet_stopwords'].str.cat(sep=' ')
    words = nltk.tokenize.word_tokenize(a)
    word_dist = nltk.FreqDist(words)
    dff2 = pd.DataFrame(word_dist.most_common(), 
                        columns=['Word', 'Frequency'])
    dff2['Word_Count'] = dff2.Word.apply(len)
    dff2[:10]    
        
    var1.set("Before the Final Most Used Words")
    label1.place(relx =0.01, rely = 0.01)
    var2.set(dff1[:10])
    label2.place(relx =0.01, rely = 0.09)   
    
    var3.set("After the Final Most Used Words")
    label3.place(relx =0.01, rely = 0.01)
    var4.set(dff2[:10])
    label4.place(relx =0.01, rely = 0.09) 

#%%

def countWords():
    global twitterDataBefore, twitterDataAfter , dataBefore , dataAfter

    dataBefore['clean_tweet_stopwords']= dataBefore['clean_tweet_stopwords'].apply(lambda x:str(x).split())
    top1 = Counter([item for sublist in dataBefore['clean_tweet_stopwords'] for item in sublist])
    temp1 = pd.DataFrame(top1.most_common(20))
    temp1.columns = ['Common_words','count']
    
    dataAfter['clean_tweet_stopwords']= dataAfter['clean_tweet_stopwords'].apply(lambda x:str(x).split())
    top2 = Counter([item for sublist in dataAfter['clean_tweet_stopwords'] for item in sublist])
    temp2 = pd.DataFrame(top2.most_common(20))
    temp2.columns = ['Common_words','count']
        
    var1.set("Before the Eurovision Final Count Common Words")
    label1.place(relx =0.01, rely = 0.01)
    var2.set(temp1.head(20))
    label2.place(relx =0.01, rely = 0.09)   
    
    var3.set("After the Eurovision Final Count Common Words")
    label3.place(relx =0.01, rely = 0.01)
    var4.set(temp2.head(20))
    label4.place(relx =0.01, rely = 0.09) 

#%%    

def hashtag():
    global twitterDataBefore, twitterDataAfter , dataBefore , dataAfter
    
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    
    hashtag_counts1 = Counter()
    for tweet in dataBefore['text']:
        hashtags1 = [tag.strip("#") for tag in tweet.split() if tag.startswith("#")]
        hashtag_counts1.update(hashtags1)
    for hashtag, count in hashtag_counts1.most_common(20):
        print("{}: {}".format(hashtag, count))
        list1.append(hashtag)
        list2.append(count)
 
        
    hashtag_counts2 = Counter()       
    for tweet in dataAfter['text']:
        hashtags2 = [tag.strip("#") for tag in tweet.split() if tag.startswith("#")]
        hashtag_counts2.update(hashtags2)
    for hashtag, count in hashtag_counts2.most_common(20):
        print("{}: {}".format(hashtag, count))
        list3.append(hashtag)
        list4.append(count)
    
    
        
    list1=pd.DataFrame(list1, columns=['Hashtags'])
    list2=pd.DataFrame(list2, columns=['Counts'])
    list3=pd.DataFrame(list3, columns=['Hashtags'])
    list4=pd.DataFrame(list4, columns=['Counts'])
    
    sonListe1=pd.concat([list1,list2], axis = 1)
    sonListe2=pd.concat([list3,list4], axis = 1)
               
    var1.set("Before the Eurovision Final Hashtag analyzing")
    label1.place(relx =0.01, rely = 0.01)
    var2.set(sonListe1)
    label2.place(relx =0.01, rely = 0.09)   
    
    var3.set("After the Eurovision Final Hashtag analyzing")
    label3.place(relx =0.01, rely = 0.01)
    var4.set(sonListe2)
    label4.place(relx =0.01, rely = 0.09) 
    
#%%

def Sentiment():
    global twitterDataBefore, twitterDataAfter , dataBefore , dataAfter, labelhashtag1, labelhashtag2, canvas

    SIA = SentimentIntensityAnalyzer()
    dataBefore["clean_tweet"]= dataBefore["clean_tweet"].astype(str)
    # Applying Model, Variable Creation
    dataBefore['Polarity Score']=dataBefore["clean_tweet"].apply(lambda x:SIA.polarity_scores(x)['compound'])
    dataBefore['Neutral Score']=dataBefore["clean_tweet"].apply(lambda x:SIA.polarity_scores(x)['neu'])
    dataBefore['Negative Score']=dataBefore["clean_tweet"].apply(lambda x:SIA.polarity_scores(x)['neg'])
    dataBefore['Positive Score']=dataBefore["clean_tweet"].apply(lambda x:SIA.polarity_scores(x)['pos'])
    # Converting 0 to 1 Decimal Score to a Categorical Variable
    dataBefore['Sentiment']=''
    dataBefore.loc[dataBefore['Polarity Score']>0,'Sentiment']='Positive'
    dataBefore.loc[dataBefore['Polarity Score']==0,'Sentiment']='Neutral'
    dataBefore.loc[dataBefore['Polarity Score']<0,'Sentiment']='Negative'    
    
    dataAfter["clean_tweet"]= dataAfter["clean_tweet"].astype(str)
    # Applying Model, Variable Creation
    dataAfter['Polarity Score']=dataAfter["clean_tweet"].apply(lambda x:SIA.polarity_scores(x)['compound'])
    dataAfter['Neutral Score']=dataAfter["clean_tweet"].apply(lambda x:SIA.polarity_scores(x)['neu'])
    dataAfter['Negative Score']=dataAfter["clean_tweet"].apply(lambda x:SIA.polarity_scores(x)['neg'])
    dataAfter['Positive Score']=dataAfter["clean_tweet"].apply(lambda x:SIA.polarity_scores(x)['pos'])
    # Converting 0 to 1 Decimal Score to a Categorical Variable
    dataAfter['Sentiment']=''
    dataAfter.loc[dataAfter['Polarity Score']>0,'Sentiment']='Positive'
    dataAfter.loc[dataAfter['Polarity Score']==0,'Sentiment']='Neutral'
    dataAfter.loc[dataAfter['Polarity Score']<0,'Sentiment']='Negative'    
        
    goster1 = twitterDataBefore.loc[:,["clean_tweet", "Sentiment"]]
    goster2= twitterDataBefore.loc[:,["clean_tweet", "Sentiment"]]
  
    
    var1.set("Before the Eurovision Final Sentiment analyzing")
    label1.place(relx =0.01, rely = 0.01)
    var2.set(goster1.head(20))
    label2.place(relx =0.01, rely = 0.09)   
    
    var3.set("After the Eurovision Final Sentiment analyzing")
    label3.place(relx =0.01, rely = 0.01)
    var4.set(goster2.head(20))
    label4.place(relx =0.01, rely = 0.09) 
    
    # Show plot

    secondWindow = tk.Toplevel()
    secondWindow.geometry("800x600") 
    secondWindow.wm_title("Sentiment Analyzing")
    
    fig = Figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    sns.countplot(x='Sentiment',data=dataAfter, ax=ax)
    
    # Figürü Tkinter penceresine yerleştirme
    canvas = FigureCanvasTkAgg(fig, master=secondWindow)
    canvas.draw()
    canvas.get_tk_widget().pack()

#%%        
# def positive():
#     global twitterDataBefore, twitterDataAfter , dataBefore , dataAfter, labelhashtag1, labelhashtag2

#     Positive_sent1 = dataBefore[dataBefore['Sentiment']=='Positive']
#     top1 = Counter([item for sublist in Positive_sent1['clean_tweet_stopwords'] for item in sublist])
#     temp_positive1 = pd.DataFrame(top1.most_common(20))
#     temp_positive1.columns = ['Common_words','count']
    
#     Positive_sent2 = dataAfter[dataAfter['Sentiment']=='Positive']
#     top2 = Counter([item for sublist in Positive_sent2['clean_tweet_stopwords'] for item in sublist])
#     temp_positive2 = pd.DataFrame(top2.most_common(20))
#     temp_positive2.columns = ['Common_words','count'] 
    
    
    
#     var1.set("Before the Final Most Common Words Positive ")
#     label1.place(relx =0.01, rely = 0.01)
#     var2.set(temp_positive1)
#     label2.place(relx =0.01, rely = 0.09)   
    
#     var3.set("After the Eurovision Final Hashtag analyzing")
#     label3.place(relx =0.01, rely = 0.01)
#     var4.set(temp_positive2)
#     label4.place(relx =0.01, rely = 0.09) 
    
# def negative ():
#     pass

# def neutral():
#     pass
#%%

def preparingData():
    global twitterDataBefore, twitterDataAfter , dataBefore , dataAfter,dataModelling, X_train, X_test, y_train, y_test,X_test_transformed
    
    dataModelling = pd.concat([dataAfter, dataBefore], ignore_index=True)
    firstData = pd.concat([dataAfter, dataBefore], ignore_index=True)
    dataModelling['Sentiment'] = dataModelling['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
    dataModelling = dataModelling.drop(["id", "created_at","clean_tweet_stopwords"], axis=1)
    stop = stopwords.words('english')
    dataModelling['clean_tweet_model'] = dataModelling['clean_tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    def cleaning_repeating_char(text):
        return re.sub(r'(.)1+', r'1', text)
    dataModelling['clean_tweet_model'] = dataModelling['clean_tweet_model'].apply(lambda x: cleaning_repeating_char(x))

    def cleaning_numbers(data):
        return re.sub('[0-9]+', '', data)
    dataModelling['clean_tweet_model'] = dataModelling['clean_tweet_model'].apply(lambda x: cleaning_numbers(x))
    
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    dataModelling['clean_tweet_model'] = dataModelling['clean_tweet_model'].apply(tokenizer.tokenize)
    
    #  Applying stemming
    import nltk
    st = nltk.PorterStemmer()
    def stemming_on_text(data):
        text = [st.stem(word) for word in data]
        return text
    dataModelling['clean_tweet_model']= dataModelling['clean_tweet_model'].apply(lambda x: stemming_on_text(x))

    # Applying lemmatizer
    import nltk
    nltk.download('wordnet')
    lm = nltk.WordNetLemmatizer()
    def lemmatizer_on_text(data):
        text = [lm.lemmatize(word) for word in data]
        return text
    dataModelling['clean_tweet_model'] = dataModelling['clean_tweet_model'].apply(lambda x: lemmatizer_on_text(x))

    # 3.1 Train - Test 
    X = dataModelling.clean_tweet_model
    y = dataModelling.Sentiment
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state =42)

    X_train_str = [' '.join(x) for x in X_train] 
    vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
    vectoriser.fit(X_train_str)
    
    X_test_str = [str(text) for text in X_test]
    X_train = vectoriser.transform(X_train_str)
    X_test_transformed = vectoriser.transform(X_test_str)
    
    goster1 = firstData.loc[:,["clean_tweet", "Sentiment"]]
    goster2= dataModelling.loc[:,["clean_tweet", "Sentiment"]]
                 
    var1.set("Before Preparing Data")
    label1.place(relx =0.01, rely = 0.01)
    var2.set(goster1.head(30))
    label2.place(relx =0.01, rely = 0.09)   
    
    var3.set("After Preparing Data")
    label3.place(relx =0.01, rely = 0.01)
    var4.set(goster2.head(30))
    label4.place(relx =0.01, rely = 0.09)      

    
    
    
def BernoulliNaiveBayesClassifier():
    global twitterDataBefore, twitterDataAfter , dataBefore , dataAfter,dataModelling, X_train, X_test, y_train, y_test,X_test_transformed

    def model_Evaluate(model):
        # Predict values for Test dataset
        y_pred = model.predict(X_test_transformed)
        # Print the evaluation metrics for the dataset.
        report = classification_report(y_test, y_pred)
        print(report)
        # Compute and plot the Confusion matrix
        cf_matrix = confusion_matrix(y_test, y_pred)
        ax = plt.subplot()
        sns.heatmap(cf_matrix, annot=True, ax=ax, cmap='Blues')
        plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
        plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
        plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
        return report, cf_matrix, y_pred
    
    BNBmodel = BernoulliNB()
    BNBmodel.fit(X_train, y_train)
    report, cf_matrix, y_pred = model_Evaluate(BNBmodel)
    
    var1.set("BernoulliNB")
    label1.place(relx=0.01, rely=0.01)
    
    var2.set(f"Classification Report: \n{report}")
    label2.place(relx=0.01, rely=0.09)   
    
    accuracy = accuracy_score(y_test, y_pred)
    var3.set(f"Accuracy: {accuracy}")
    label3.place(relx=0.01, rely=0.19)
    var4.set("")
    label4.place(relx =0.01, rely = 0.09)
    
    ax = plt.subplot()
    sns.heatmap(cf_matrix, annot=True, ax=ax, cmap='Blues')
    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
    plt.show()




    
def SVM():
    global twitterDataBefore, twitterDataAfter , dataBefore , dataAfter,dataModelling, X_train, X_test, y_train, y_test,X_test_transformed

    def model_Evaluate(model):
        # Predict values for Test dataset
        y_pred = model.predict(X_test_transformed)
        # Print the evaluation metrics for the dataset.
        report = classification_report(y_test, y_pred)
        print(report)
        # Compute and plot the Confusion matrix
        cf_matrix = confusion_matrix(y_test, y_pred)
        ax = plt.subplot()
        sns.heatmap(cf_matrix, annot=True, ax=ax, cmap='Blues')
        plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
        plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
        plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
        return report, cf_matrix, y_pred
    
    SVCmodel = LinearSVC()
    SVCmodel.fit(X_train, y_train)
    report, cf_matrix, y_pred = model_Evaluate(SVCmodel)
    
    var1.set("SVM (Support Vector Machine)")
    label1.place(relx=0.01, rely=0.01)
    
    var2.set(f"Classification Report: \n{report}")
    label2.place(relx=0.01, rely=0.09)   
    
    accuracy = accuracy_score(y_test, y_pred)
    var3.set(f"Accuracy: {accuracy}")
    label3.place(relx=0.01, rely=0.19)
    var4.set("")
    label4.place(relx =0.01, rely = 0.09)
    
    ax = plt.subplot()
    sns.heatmap(cf_matrix, annot=True, ax=ax, cmap='Blues')
    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
    plt.show()
        

def Logistic():
    global twitterDataBefore, twitterDataAfter , dataBefore , dataAfter,dataModelling, X_train, X_test, y_train, y_test,X_test_transformed

    def model_Evaluate(model):
        # Predict values for Test dataset
        y_pred = model.predict(X_test_transformed)
        # Print the evaluation metrics for the dataset.
        report = classification_report(y_test, y_pred)
        print(report)
        # Compute and plot the Confusion matrix
        cf_matrix = confusion_matrix(y_test, y_pred)
        ax = plt.subplot()
        sns.heatmap(cf_matrix, annot=True, ax=ax, cmap='Blues')
        plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
        plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
        plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
        return report, cf_matrix, y_pred
    
    LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
    LRmodel.fit(X_train, y_train)
    report, cf_matrix, y_pred = model_Evaluate(LRmodel)
    
    var1.set("Logistic Regression")
    label1.place(relx=0.01, rely=0.01)
    
    var2.set(f"Classification Report: \n{report}")
    label2.place(relx=0.01, rely=0.09)   
    
    accuracy = accuracy_score(y_test, y_pred)
    var3.set(f"Accuracy: {accuracy}")
    label3.place(relx=0.01, rely=0.19)
    var4.set("")
    label4.place(relx =0.01, rely = 0.09)
    
    ax = plt.subplot()
    sns.heatmap(cf_matrix, annot=True, ax=ax, cmap='Blues')
    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
    plt.show()


#%%
    
"menübar ve butonlarını oluşturma" 

def close_window():
    window.destroy()
    
menubar = tk.Menu (window)
window.config(menu = menubar)
file = tk.Menu(menubar, tearoff=0) # tearoff seçenekleri en üstten koymaya başlamamızı sağlar
modelling = tk.Menu(menubar, tearoff=0)
analiz = tk.Menu(file, tearoff=0)
sentiment= tk.Menu(analiz, tearoff = 0)

"add_cascade ile menübar üzerine yeni bir parent butonları ekliyoruz"

menubar.add_cascade (label="Data", menu = file) 
menubar.add_cascade (label="Modelling" , menu = modelling)
modelling.add_cascade(label="Analysis of Tweets-3", menu =analiz)
# analiz.add_cascade(label="Sentiment of Tweets", menu =sentiment)

" add_command ile alt menüleri oluşturuyoruz"

#file menüsünün alt menülerini oluşturuyoruz

file.add_command (label = "Open -1", command = OpenData)
file.add_command (label = "Data First Cleaning- 2", command = DataFirstCleaning)
file.add_separator()  # menü seçenekleri arasına bir ayırıcı çizgi ekler
file.add_command (label = "Exit", command = close_window ) 


# patient info alt menülerini oluşturuyoruz 
modelling.add_command (label = "Preparing Data -8", command = preparingData)
modelling.add_command (label = "Bernoulli Naive Bayes Classifier -9", command = BernoulliNaiveBayesClassifier)
modelling.add_command (label = "SVM (Support Vector Machine) -10", command = SVM) 
modelling.add_command (label = "Logistic Regression - 11", command = Logistic) 

analiz.add_command(label='Most Used Words -4', command = usedWords)
analiz.add_command(label='Count the common words- 5', command = countWords)
analiz.add_command(label='Hashtag Analyzing -6', command = hashtag)
analiz.add_command(label='Sentiment Analyzing- 7', command = Sentiment)

window.mainloop() 