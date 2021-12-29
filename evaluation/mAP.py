import numpy as np
from progress.bar import Bar


def hamming_distance(vector1, matrix):
    distance = np.sum(vector1 != matrix, axis=1)
    return distance


def ecul_distance(vector1, matrix):
    distance = np.sum(np.sqrt(np.square(matrix - vector1)),
                      axis=1)
    return distance


def cal_AP(query, database, top_k, distance_type="hamming"):
    query_data, query_label = query[0], query[1]
    target_data, target_label = database[0], database[1]
    query_data = np.reshape(query_data, (1, query_data.shape[0]))
    target_label = np.reshape(target_label, (target_label.shape[0]))
    precision_list = []
    if distance_type == "hamming":
        distance_matrix = hamming_distance(query_data, target_data)
    elif distance_type == "ecul":
        distance_matrix = ecul_distance(query_data, target_data)
    else:
        raise("distance type is error")

    id = np.argsort(distance_matrix, axis=0)[:top_k]
    for i in range(top_k):
        ranked_result_labels = target_label[id[:i + 1]]
        precision = np.mean(np.equal(ranked_result_labels, query_label))
        precision_list.append(precision)
    ap = np.mean(np.array(precision_list))
    return ap


def cal_mAP(query, database, with_top=20, distance_type="hamming"):
    query_set = query[0]
    query_set_label = query[1]
    AP_list = []
    bar = Bar('calculating mAP', max=query_set.shape[0])
    for i in range(0, query_set.shape[0]):
        current_query = [query_set[i], query_set_label[i]]
        current_query_AP = cal_AP(
            current_query, database, with_top, distance_type=distance_type)
        AP_list.append(current_query_AP)
        temp_value = np.mean(np.array(AP_list))
        bar.suffix = 'value :{value:.4f}'.format(value=temp_value)
        bar.next()
    bar.finish()
    mAP = np.mean(np.array(AP_list))
    return mAP
