from flask import Flask, render_template, flash, redirect, url_for, request
from wtforms import Form, StringField, validators
from csv import reader
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from math import log10, sqrt
from collections import OrderedDict
import string
from nltk.corpus import stopwords
import random
import time
import pickle

def remove(s):
	return "".join(s.split())


all_shingles = []
hash_functions = []
stop_words = set(stopwords.words('english'))
final_query_matrix = []
final_matrix = []
num_hash_values = 100
shingle_doc_matrix = []
num_docs = 1000
num_shingles = 0

def get_all_shingles():
	print("Running get_all_shingles for: " + str(num_docs) + " documents")
	start_time = time.time()
	global num_shingles
	f = open('bible_data_set_1.csv', 'r')

	k = reader(f)

	l = 0


	for i in k:
		l = l+1

		#terms_list = list(map(lambda x: PorterStemmer().stem(x), word_tokenize(i[4] + i[1])))
		real_terms_list = word_tokenize(i[4])

		filtered_list = []
		for w in real_terms_list:
			w = w.lower()
			if w not in stop_words and w not in string.punctuation and w not in ["-"]:
				filtered_list.append(w)
		sentence = "".join(filtered_list)

		for j in range(0, len(sentence)-2):
			shingle = sentence[j:j+3]
			if shingle not in all_shingles:
				all_shingles.append(shingle)


		if(l == num_docs):
			break
	num_shingles = len(all_shingles)
	all_shingles.sort()
	with open("all_shingles_" + str(num_hash_values) + ".pickle", "wb") as handle:
		pickle.dump(all_shingles, handle, protocol=pickle.HIGHEST_PROTOCOL)
	handle.close()
	f.close()
	curr_time = time.time()
	print("Running time for get_all_shingles: " + str(curr_time-start_time) + " secs")


def create_matrix():

	print("Running create_matrix for: " + str(num_docs) + " documents")
	start_time = time.time()

	f = open('bible_data_set_1.csv', 'r')

	k =list( reader(f) )
	
	for j in all_shingles:
		shingle_doc_matrix.append([])

	l=0
	pre_store = []

	for s in range(0, num_docs):
		doc = k[s][4]
		real_terms_list = word_tokenize(doc)
		filtered_list = []
		for w in real_terms_list:
			w = w.lower()
			if w not in stop_words and w not in string.punctuation and w not in ["-"]:
				filtered_list.append(w)
		sentence = "".join(filtered_list)
		shingle_list_doc = set()
		for t in range(0, len(sentence) - 2):
			shingle = sentence[t:t+3]
			shingle_list_doc.add(shingle)

		pre_store.append(shingle_list_doc)

	for j in all_shingles:
		new_shingle = all_shingles[l]
		doc_size = 0
		for i in range(0,num_docs):
			set_doc = pre_store[i]
			if(new_shingle not in set_doc):
				shingle_doc_matrix[l].append(0)
			else:
				shingle_doc_matrix[l].append(1)

		l = l+1
	with open("shingle_doc_matrix_" + str(num_hash_values) + ".pickle", "wb") as handle:
		pickle.dump(shingle_doc_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
	handle.close()
	curr_time = time.time()
	print("Running time for create_matrix: " + str(curr_time-start_time) + " secs")


def create_k_signature():
	
	print("Running create_k_signature for: " + str(num_docs) + " documents")
	start_time = time.time()
	for i in range(0, num_hash_values):
		a = random.randint(1, num_shingles-1)
		b = random.randint(1, num_shingles-1)
		hash_functions.append([a,b])

	with open("hash_functions_" + str(num_hash_values) + ".pickle", "wb") as handle:
		pickle.dump(hash_functions, handle, protocol=pickle.HIGHEST_PROTOCOL)
	handle.close()

	for j in range(0, num_hash_values):
		final_matrix.append([])
		for i in range(0, num_docs):
			final_matrix[j].append(99999)


	for i in range(0, num_shingles):

		for j in range(0, num_docs):
#            print(i, j);
			if shingle_doc_matrix[i][j] == 1:
				vals = []
				for k in range(0, num_hash_values):
					vals.append((hash_functions[k][0]*i+hash_functions[k][1])%(num_shingles) )
				for k in range(0, num_hash_values):
					if (final_matrix[k][j] > vals[k]):
						final_matrix[k][j] = vals[k]
	curr_time = time.time()
	print("Running time for create_k_signature: " + str(curr_time-start_time) + " secs")

def pre_processing(hash_values_cnt):
	global num_hash_values
	num_hash_values = hash_values_cnt
	get_all_shingles()
	
	create_matrix()
	create_k_signature()
	with open("signature_matrix_" + str(num_hash_values) + ".pickle", "wb") as handle:
		pickle.dump(final_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
	handle.close()


def query_signature_matrix(q,all_shingles,hash_functions):

	# with open("all_shingles_" + str(num_hash_values) + ".pickle", "rb") as handle:
	# 	all_shingles = pickle.load(handle)
	# handle.close()
	#
	# with open("hash_functions_" + str(num_hash_values) + ".pickle", "rb") as handle:
	# 	hash_functions = pickle.load(handle)
	# handle.close()

	#print(all_shingles)
	real_terms_list = word_tokenize(q)
	filtered_list = []
	for w in real_terms_list:
		w=w.lower()
		if w not in stop_words and w not in string.punctuation and w not in ["-"]:
			filtered_list.append(w)

	sentence = "".join(filtered_list)
	shingles_query = []
	for t in range(0, len(sentence) - 2):
		shingle = sentence[t:t+3]
		shingles_query.append(shingle)
	shingle_query_matrix = []
	num_shingles = len(all_shingles)
	for i in range(0, num_shingles):
		shingle_query_matrix.append([])


	for i in range(0, num_shingles):
		new_shingle = all_shingles[i]
		if new_shingle in shingles_query:
			shingle_query_matrix[i] = 1
		else:
			shingle_query_matrix[i]= 0

	signature_query = []
	for i in range(0, num_hash_values):
		signature_query.append(99999)
	#K is 6
	for i in range(0, num_shingles):
		if(shingle_query_matrix[i] == 1):

			vals = []
			for k in range(0, num_hash_values):
				vals.append((hash_functions[k][0]*i+hash_functions[k][1])%(num_shingles))
			for k in range(0, num_hash_values):
				if(signature_query[k] > vals[k]):
					signature_query[k] = vals[k]

	final_query_matrix = signature_query
	'''
		final_query_matrix		it is the signature matrix of the query which is a list
		shingle_query_matrix	it is the shingle to query matrix, it is a list
	'''

	return final_query_matrix, shingle_query_matrix

# if __name__ == "__main__":
# 	pre_processing(100)


# if __name__ ==  '__main__':
#     curr_time = time.time()
#     get_all_shingles()
#     all_shingles.sort()
#     print("All shingles collected and sorted")
#     print("Time taken: " + str(time.time() - curr_time) + " secs")
#     curr_time = time.time()
#     create_matrix()
#     print("Big matrix created")
#     print("Time taken: " + str(time.time() - curr_time) + " secs")
#     curr_time = time.time()
#     create_k_signature()
#     print("signature matrix created" + " for " + str(num_hash_values) + " hash functions")
#     print("Time taken: " + str(time.time() - curr_time) + " secs")
#     curr_time = time.time()
#     with open("signature_matrix" + ".pickle", "wb") as handle:
#         pickle.dump(final_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     handle.close()
#     print("Wrote into pickle files")
#     print("Time taken: " + str(time.time() - curr_time) + " secs")




