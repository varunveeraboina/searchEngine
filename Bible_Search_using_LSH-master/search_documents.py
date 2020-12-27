import math
import numpy as np
from scipy.spatial import distance
'''
functionality:
    This function give the candidate documents that a given query document is related to
arguments:
    buckets_list: this is a list of list of lists, where each has the doc-ids of documents which are possible candidates
    [
        [
            [0, 1], [2, 3], [4, 5, 6]   #this means 0, 1 got hashed into the first bucket in the first band
        ], 
        [
            [], [], []
        ]
        , 
        [
            [], [], []
        ]
    ]
    query_bucket_list: this is a list, where each element is the bucket number to which the query got hashed in different bands
    [
        2, 1, 3, 4      #this means the query got hashed to 2nd bucket in the first band
    ]
return:
    doc_list: this is a list of doc-ids which are possible candidates to the query document
'''
def getDocuments(buckets_list, query_bucket_list):
    
    doc_list = list()
    num_bands = len(query_bucket_list)
    for band in range(num_bands):
        bucket_index = query_bucket_list[band]
        docs = buckets_list[band][bucket_index]
        for doc in docs:
            if doc not in doc_list:
                doc_list.append(doc)
    return doc_list

'''
functionality:
    This function validates the list of candidate documents given by LSH and returns a list of documents which are actually similar
arguments:
    signature_matrix: The matrix where each column is a document and each row is a signature value obtained after min-hashing
        d1  d2  d3
    s1  0   1   1   
    s2  1   0   1   
    s3  1   0   0
    query_signature: It is a list of lists, where each list is a signature value of the query
    [
        [1],    #This is the s1 value of the query
        [0],
        [1]
    ]
    candidate_documents: (list of doc-ids)This is a list of all the documents which are possible candidates as given by LSH algorithm
    threshold: (float value)The threshold value which is essentially (1/b)^(1/r), where b = num of bands, r = num of signatures in each band
    similarity_measure: (string)The similarity measure which should be applied to validate a candidate document
        "euclidean"         : Euclidean distance is taken between two documents
        "jaccard"           : Jaccard distance is taken between two documents
        "cosine"            : Cosine distance is taken between two documents
        "editdistance"      : Edit distance is taken between two documents
        "hammingdistance"   : Hamming distance is taken between two documents
return:
    doc_list: this is a list of doc-ids which are validated similar documents
'''


def Sort_Tuple(tup):
    tup.sort(key=lambda x: x[1], reverse= False)
    return tup
def getValidatedDocuments(signature_matrix, query_signature, candidate_documents, threshold, similarity_measure):

    doc_similarity_list=list()
    for doc in candidate_documents:
        doc_signature = getSignature(doc, signature_matrix)
        similarity_value = findSimilarity(doc_signature, query_signature, similarity_measure)
        #similarity_value = 10
        doc_similarity_list.append((doc, similarity_value))
        # if similarity_value >= threshold:
        #     if doc not in doc_list:
        #         doc_list.append(doc)
    doc_similarity_list = Sort_Tuple(doc_similarity_list)
    final_list= []
    for i in range(min(10,len(doc_similarity_list))):
        final_list.append(doc_similarity_list[i][0])
    return final_list

'''
functionality:
    This is a utility function which extracts gives the signature of a document using the signature matrix
arguments:
    doc: This is the doc-id of the document
    signature_matrix: The matrix where each column is a document and each row is a signature value obtained after min-hashing
        d1  d2  d3
    s1  0   1   1   
    s2  1   0   1   
    s3  1   0   0
return:
    doc_signature: It is a list of lists, where each list is a signature value of the query
    [
        [1],    #This is the s1 value of the query
        [0],
        [1]
    ]
'''

def getSignature(doc, signature_matrix):


    # doc_signature = list()
    # for signature in range(len(signature_matrix)):
    #     doc_signature.append([signature_matrix[signature][doc]])
    #doc_signature = numpy.array(signature_matrix)
    doc_signature= signature_matrix[:,doc]
    return doc_signature


'''
functionality:
    This function returns the similarity value measure given two document signatures
arguments:
    doc1_signature: It is a list of lists, where each list is a signature value of the query
    [
        [1],    #This is the s1 value of the query
        [0],
        [1]
    ]
    doc2_signature: It is a list of lists, where each list is a signature value of the query
    [
        [1],    #This is the s1 value of the query
        [0],
        [1]
    ]
    similarity_measure: (string)The similarity measure which should be applied to validate a candidate document
        "euclidean"         : Euclidean distance is taken between two documents
        "jaccard"           : Jaccard distance is taken between two documents
        "cosine"            : Cosine distance is taken between two documents
        "editdistance"      : Edit distance is taken between two documents
        "hammingdistance"   : Hamming distance is taken between two documents
return:
    similarity_value: (float) This is the similarity_value between the given two documents
'''
def findSimilarity(doc1_vector, doc2_vector, similarity_measure):

    # doc1_vector = []
    # doc2_vector = []
    # num_signature = len(doc1_signature)
    #
    # for signature in range(num_signature):
    #     doc1_vector.append(doc1_signature[signature][0])
    #     doc2_vector.append(doc2_signature[signature][0])
    similarity_value = distance.jaccard(doc1_vector,doc2_vector)
    #similarity_value = 1 - distance.jaccard(doc1_vector, doc2_vector)
    #similarity_value = jaccard_measure(doc1_vector, doc2_vector)
    # if similarity_measure=="euclidean":
    #     similarity_value = euclidean_measure(doc1_vector, doc2_vector)
    # elif similarity_measure=="jaccard":
    #     similarity_value = jaccard_measure(doc1_vector, doc2_vector)
    # elif similarity_measure=="cosine":
    #     similarity_value = cosine_measure(doc1_vector, doc2_vector)
    # elif similarity_measure=="editdistance":
    #     similarity_value = editdistance_measure(doc1_vector, doc2_vector)
    # elif similarity_measure=="hammingdistance":
    #     similarity_value = hammingdistance_measure(doc1_vector, doc2_vector)
    #
    return similarity_value

'''
functionality: This function computes the euclidean distance measure between two documents
arguments:
    doc1_vector: This is a list of numbers representing the position of the document in d-dimensions, where d=len of the list
    doc1_vector: This is a list of numbers representing the position of the document in d-dimensions, where d=len of the list
return:
    measure: this is the euclidean measure between the two documents
'''
def euclidean_measure(doc1_vector, doc2_vector):
    
    num_dimensions = len(doc1_vector)
    measure = 0
    for dimension in range(num_dimensions):
        measure += (doc1_vector[dimension] - doc2_vector[dimension])**2
    measure = math.sqrt(measure)
    return measure

'''
functionality: This function computes the jaccard similarity measure between two documents
arguments:
    doc1_vector: This is a list of numbers representing the position of the document in d-dimensions, where d=len of the list
    doc1_vector: This is a list of numbers representing the position of the document in d-dimensions, where d=len of the list
return:
    measure: this is the jaccard similarity measure between the two documents
'''
def jaccard_measure(doc1_vector, doc2_vector):
    num_dimensions = len(doc1_vector)
    intersection_cnt = 0
    union_cnt = 0
    for dimension in range(num_dimensions):
        if doc1_vector[dimension] + doc2_vector[dimension] == 0:
            continue
        elif doc1_vector[dimension] * doc2_vector[dimension] == 1:
            intersection_cnt += 1
            union_cnt += 1
        else:
            union_cnt += 1
    measure = intersection_cnt / union_cnt
    return measure

'''
functionality: 
    This function computes the cosine distance measure between two documents
arguments:
    doc1_vector: This is a list of numbers representing the position of the document in d-dimensions, where d=len of the list
    doc1_vector: This is a list of numbers representing the position of the document in d-dimensions, where d=len of the list
return:
    measure: this is the cosine measure between the two documents
'''
def cosine_measure(doc1_vector, doc2_vector):

    dot_product = 0
    doc1_norm = 0
    doc2_norm = 0
    num_dimensions = len(doc1_vector)
    for dimension in range(num_dimensions):
        dot_product += doc1_vector[dimension]*doc1_vector[dimension]
        doc1_norm += (doc1_vector[dimension])**2
        doc2_norm += (doc2_vector[dimension])**2
    
    doc1_norm = math.sqrt(doc1_norm)
    doc2_norm = math.sqrt(doc2_norm)
    measure = dot_product/(doc1_norm*doc2_norm)
    return measure


'''
functionality:  
    This function computes the edit distance measure between two documents
arguments:
    doc1_vector: This is a list of numbers representing the position of the document in d-dimensions, where d=len of the list
    doc1_vector: This is a list of numbers representing the position of the document in d-dimensions, where d=len of the list
return:
    measure: this is the edit distance measure between the two documents
'''
def editdistance_measure(doc1_vector, doc2_vector):
    
    num_dimensions = len(doc1_vector)
    doc1_str = ""
    doc2_str = ""
    for dimension in range(num_dimensions):
        doc1_str += str(doc1_vector[dimension])
        doc2_str += str(doc2_vector[dimension])
    measure = editDistance(doc1_str, doc2_str, len(doc1_str), len(doc2_str))
    return measure
    
'''
functionality:
    This is a utility function which calculates the edit distance between two strings
arguments:
    str1: First string
    str2: Second string
    m: length of first string
    n: length of second string
return:
    returns the edit distance between the strings
'''
def editDistance(str1, str2, m , n): 
    
    if m==0: 
         return n 
    if n==0: 
        return m  
    if str1[m-1]==str2[n-1]: 
        return editDistance(str1,str2,m-1,n-1) 
    return 1 + min(editDistance(str1, str2, m, n-1), editDistance(str1, str2, m-1, n), editDistance(str1, str2, m-1, n-1)) 


'''
functionality:  
    This function computes the hamming distance measure between two documents
arguments:
    doc1_vector: This is a list of numbers representing the position of the document in d-dimensions, where d=len of the list
    doc1_vector: This is a list of numbers representing the position of the document in d-dimensions, where d=len of the list
return:
    measure: this is the hamming distance measure between the two documents
'''
def hammingdistance_measure(doc1_vector, doc2_vector):
    
    num_dimensions = len(doc1_vector)
    measure = 0
    for dimension in range(num_dimensions):
        if doc1_vector[dimension] != doc2_vector[dimension]:
            measure += 1

    return measure


