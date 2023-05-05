from flask import Flask, request, render_template ,redirect
import string
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

lr = "static/models/lr_model.pkl"
model = pickle.load(open(lr,'rb'))
result =[]
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

STOPWORDS = set(stopwordlist)
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


english_punctuations = string.punctuation
punctuations_list = english_punctuations

def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)

def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)

def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)

st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data

lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data

app = Flask(__name__)
file = open("static/js/temp.js",'w')
file.write(f'var result = ""')

@app.route('/',methods=['POST','GET'])
def index():
    if request.method == "POST":
        data = request.form.get("input")
        result = data
        data = cleaning_stopwords(data)
        data = cleaning_repeating_char(data)
        data = cleaning_URLs(data)
        data = cleaning_numbers(data)
        data = data.split(" ")
        data = stemming_on_text(data)
        data = lemmatizer_on_text(data)



        vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
        vectoriser=vectoriser.fit(data)


        data  = vectoriser.transform(data)

        print(model.predict(data))

        
        file = open("static/js/temp.js",'w')
        file.write(f'var result = "{result}"')

    return render_template('home.html')

app.ddebug = True
app.run(host='0.0.0.0',port = 5000)


    






