import random
import numpy as np
from cka_cal import *


def get_init_centers(raw_data, k):
    return random.sample(raw_data, k)


def cal_distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

def get_cluster_with_mse(raw_data, centers):
    distance_sum = 0.0
    cluster = {}
    for item in raw_data:
        flag = -1
        min_dis = float('inf')
        for i, center_point in enumerate(centers):
            dis = cal_distance(item, center_point)
            if dis < min_dis:
                flag = i
                min_dis = dis
        if flag not in cluster:
            cluster[flag] = []
        cluster[flag].append(item)
        distance_sum += min_dis**2
    return cluster, distance_sum/(len(raw_data)-len(centers))


def get_new_centers(cluster):
    center_points = []
    for key in cluster.keys():
        center_points.append(np.mean(cluster[key], axis=0)) 
    return center_points


def k_means(raw_data, k, mse_limit, early_stopping):
    old_centers = get_init_centers(raw_data, k)
    old_cluster, old_mse = get_cluster_with_mse(raw_data, old_centers)
    new_mse = 0
    count = 0
    while np.abs(old_mse - new_mse) > mse_limit and count < early_stopping : 
        old_mse = new_mse
        new_center = get_new_centers(old_cluster)

        new_cluster, new_mse = get_cluster_with_mse(raw_data, new_center)  
        count += 1

    return new_cluster

