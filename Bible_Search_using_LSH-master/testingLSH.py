from LSH import LSH
import search_documents as sd
import functions as fun
import pickle
import numpy as np

if __name__ == '__main__':


    
    num_bands = 10
    buckets_count = 10
    num_hash_values = 100
    #fun.pre_processing(num_hash_values)
    
    r_band = num_hash_values/num_bands
    threshold = (1/num_bands)**(1/r_band)
    
    with open("shingle_doc_matrix_" + str(num_hash_values) + ".pickle", "rb") as handle:
        shingling_matrix = pickle.load(handle)
        shingling_matrix = np.array(shingling_matrix)
    handle.close()

    with open("signature_matrix_" + str(num_hash_values) + ".pickle", "rb") as handle:
        signature_matrix = pickle.load(handle)
    handle.close()
    
    with open("all_shingles_" + str(num_hash_values) + ".pickle", "rb") as handle:
        all_shingles = pickle.load(handle)
    handle.close()

    with open("hash_functions_" + str(num_hash_values) + ".pickle", "rb") as handle:
        hash_functions = pickle.load(handle)
    handle.close()

    #list = [[1,0,1],[1,1,1],[1,0,0],[0,0,1],[0,1,0],[0,1,1]]
    #search_list = [[1],[0],[1],[0],[1],[1]]
    lsh = LSH(signature_matrix,num_bands,buckets_count)

    buckets_list = lsh._bucket_matrix()
    #print(buckets_list[1])
    #print(len(buckets_list),len(buckets_list[0]))
    #lsh_search = LSH(search_list,3,2,True)
    #query_bucket_list = lsh_search._bucket_value_matrix()
    #print(query_bucket_list)
    #doc_list = sd.getDocuments(buckets_list, query_bucket_list)
    #print(doc_list)
    #print(all_shingles)
    query_string = input("Enter the query: ")
    search_list,query_shingling_matrix = fun.query_signature_matrix(query_string,all_shingles,hash_functions)

    query_shingling_matrix = np.reshape(query_shingling_matrix,(len(query_shingling_matrix),1))
    
    lsh_search = LSH(search_list,num_bands,buckets_count,True)
    query_bucket_list = lsh_search._bucket_value_matrix()
    doc_list = sd.getDocuments(buckets_list, query_bucket_list)
    #print(query_shingling_matrix)
    print(len(doc_list))
    print(threshold)
    doc_list = sd.getValidatedDocuments(shingling_matrix,query_shingling_matrix,doc_list,threshold,"jaccard")
    print(len(doc_list))
    
    
