import pickle

from flask import Flask, render_template, flash, redirect, url_for, request
from wtforms import Form, StringField, validators
from csv import reader
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from math import log10, sqrt
from collections import OrderedDict
from LSH import LSH
import search_documents as sd
import functions as fun
import pickle
import numpy as np
import time
app = Flask(__name__)





def readData():
    '''to read data from the dataset and load it into Data Structures'''
    f = open('bible_data_set_1.csv', 'r')
    k = reader(f)
    l = 0
    doc_names = []
    for i in k:
        l += 1
        doc_name = i[0]
        doc_names.append(doc_name)

    f.close()
    return doc_names

def getDocDetails(docname):
    '''returns the document with the (docname)'''
    f = open('bible_data_set_1.csv', 'r')

    k = reader(f)

    l = 0
    for i in k:
        #print(l)
        l += 1
        doc_name = i[0]
        if docname == doc_name:
            f.close()
            return i[0], i[1], i[2], i[3], i[4]


class SearchForm(Form):
    '''Form class to support WTForms in Flask'''
    search = StringField('Search for...', [validators.InputRequired()])


@app.route('/searchResults/<string:query>', methods=['GET', 'POST'])
def searchResults(query):
    '''returns the page with the top 10 relevant search results'''
    print("Running for query: " + query)
    start_time = time.time()
    search_list,query_shingling_matrix = fun.query_signature_matrix(query, all_shingles, hash_functions)
    query_shingling_matrix = np.reshape(query_shingling_matrix, (len(query_shingling_matrix), 1))
    lsh_search = LSH(search_list, band_value, buckets_count, True)
    query_bucket_list = lsh_search._bucket_value_matrix()
    doc_list = sd.getDocuments(buckets_list, query_bucket_list)
    #print(len(doc_list))
    doc_list = sd.getValidatedDocuments(shingling_matrix, query_shingling_matrix, doc_list, 0.5, "jaccard")
    curr_time = time.time()
    print("Running time for query: " + str(curr_time-start_time) + " secs")
    results = [doc_names[doc_id] for doc_id in doc_list]
    form = SearchForm(request.form)
    if request.method == 'POST' and form.validate():
        search = form.search.data
        return redirect(url_for('searchResults', query=search))
    # if query != real_search:
    #     flash('Showing results for ' + real_search, 'success')
    return render_template('searchResults.html', results=results, form=form)


@app.route('/displayDoc/<string:docname>')
def displayDoc(docname):
    '''returns the document with given docname'''
    citation, book, chapter, verse, text = getDocDetails(docname)
    return render_template('document.html', citation=citation, book=book, chapter=chapter, verse=verse, text=text)


@app.route('/', methods=['GET', 'POST'])
def index():
    '''returns the home page with search bar'''
    form = SearchForm(request.form)
    if request.method == 'POST' and form.validate():
        search = form.search.data
        return redirect(url_for('searchResults', query=search, form=form))
    return render_template("home.html", form=form)


if __name__ == '__main__':
    doc_names = readData()
    band_value = 20
    buckets_count = 20
    # fun.pre_processing()
    num_hash_values = 100

    # Initially run fun.preprocessing() to generate this shingle_doc_matrix
    with open("shingle_doc_matrix_" + str(num_hash_values) + ".pickle", "rb") as handle:
        shingling_matrix = pickle.load(handle)
        shingling_matrix = np.array(shingling_matrix)
    handle.close()
      
    with open("signature_matrix_" + str(num_hash_values) + ".pickle", "rb") as handle:
        signature_matrix = pickle.load(handle)
    handle.close()

    # list = [[1,0,1],[1,1,1],[1,0,0],[0,0,1],[0,1,0],[0,1,1]]
    # search_list = [[1],[0],[1],[0],[1],[1]]
    lsh = LSH(signature_matrix, band_value, buckets_count)

    buckets_list = lsh._bucket_matrix()

    with open("all_shingles_" + str(num_hash_values) + ".pickle", "rb") as handle:
        all_shingles = pickle.load(handle)
    handle.close()
    with open("hash_functions_" + str(num_hash_values) + ".pickle", "rb") as handle:
        hash_functions = pickle.load(handle)
    handle.close()

    app.secret_key = '528491@JOKER'
    app.run()