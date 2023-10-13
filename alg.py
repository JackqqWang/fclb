import seaborn as sns
import os
import tqdm
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import copy
# import time
from tools_cnn_handcraft import *
from server import *
from models import *
from tools_cnn_handcraft import *
from server_utils import *
from cluster_alg import *
from options import args_parser
from utils import exp_details, get_datasets, get_pub_datasets, get_public_datasets, average_weights
from update import LocalUpdate

from test import test_inference
from local_tools import *
from sampling import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    path_project = os.path.abspath('..')
     
    exp_details(args)
    device = args.device
    
    # server_train_data = get_server_train(dataset, args)

    train_dataset, test_dataset, dict_users_train, dict_users_test, server_train_data = get_datasets(args)

    if args.public_dataset == args.dataset:
        data_at_server = server_train_data
    else:
        data_at_server = get_public_datasets(args)
    if args.public_dataset == 'cifar10':
        input_size = 32 * 32 * 3
    elif args.public_dataset == 'mnist':
        input_size = 32 * 32 * 3
    elif args.public_dataset == 'svhn':
        input_size = 32 * 32 * 3
    else:
        print('wrong public dataset name')

    if args.model_same == 1:
        if args.model == 'CNN1':
            model_indicator = 'A'
        elif args.model == 'CNN2':
            model_indicator = 'B'
        elif args.model == 'CNN3':
            model_indicator = 'C'
        elif args.model == 'CNN4':
            model_indicator = 'D'
        else:
            print('wrong model name')
        client_model_dict = {}
        for client_idx in range(args.num_users):
            client_model_dict[client_idx] = model_indicator
   
    client_model_dict = model_generation(args.num_users) 

    local_avg_train_losses_list, local_avg_train_accuracy_list = [],[]
    local_avg_test_losses_list, local_avg_test_acc_list = [], []
    local_avg_test_accuracy_list = []
    print_every = 1
    

    model_assign_dict, client_id_model_name = model_assign_generator(args.num_users)
    global_models = {}
    global_models['CNN1'] = CNN1(args=args).to(device)
    global_models['CNN2'] = CNN2(args=args).to(device)
    global_models['CNN3'] = CNN3(args=args).to(device)
    global_models['CNN4'] = CNN4(args=args).to(device)

    print("start server and client communication:")
    previous_user_list = []
    current_user_id_modelweights_dict = {}
    for epoch in tqdm(range(args.epochs)):


        local_losses = []


        local_test_losses, local_test_accuracy = [],[]
        print('communication round: {} \n'.format(epoch))


        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace = False)

        local_weights = {}
        selected_names = [client_id_model_name[id] for id in idxs_users]
        for model_name in list(set(selected_names)):
            local_weights[model_name] = []

        for idx in idxs_users:
            test_loader_for_each_client = torch.utils.data.DataLoader(
                dataset=DatasetSplit(train_dataset, dict_users_test[idx]),
                shuffle=True,
            )
            local_model = LocalUpdate(args = args, dataset = train_dataset, idxs = dict_users_train[idx])

   
            if epoch == 0:
                w, loss = local_model.update_weights(model = copy.deepcopy(model_assign_dict[idx]), global_round = epoch)
                trained_local_model = copy.deepcopy(model_assign_dict[idx])
                trained_local_model.load_state_dict(w)
                trained_local_model.to(device)
                test_acc, test_loss = test_inference(args, trained_local_model, test_loader_for_each_client)
            
            
            else:
                
                if idx in previous_user_list:
                    #TODO pull the idx model down to the local
                    teacher_model = best_model_dict[idx]
                    currrent_student_model = current_user_id_modelweights_dict[idx]
                    w, loss = local_model.k_distll(student_model = currrent_student_model, teacher_model = teacher_model)
                    trained_local_model = copy.deepcopy(current_user_id_modelweights_dict[idx])
                    trained_local_model.load_state_dict(w)
                    test_acc, test_loss = test_inference(args, trained_local_model, test_loader_for_each_client)

                else:
                    w, loss = local_model.update_weights(model = copy.deepcopy(model_assign_dict[idx]), global_round = epoch)
                    trained_local_model = copy.deepcopy(model_assign_dict[idx])
                    trained_local_model.load_state_dict(w)
                    test_acc, test_loss = test_inference(args, trained_local_model, test_loader_for_each_client)

            local_weights[client_id_model_name[idx]].append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss.detach().cpu().item()))
            local_test_losses.append(test_loss) # len = user number 
            local_test_accuracy.append(test_acc)
            temp_temp_model = copy.deepcopy(model_assign_dict[idx])
            temp_temp_model.load_state_dict(w)
            # local_model_list.append(temp_temp_model)
            current_user_id_modelweights_dict[idx] = temp_temp_model

        selected_models = {}
        index = 0
        for k, v in local_weights.items():
            print(len(v))
            global_weights = average_weights(v)
            global_models[k].load_state_dict(global_weights)
            selected_models[index] = global_models[k]
            index += 1

        loss_avg = sum(local_losses) / len(local_losses)
        loss_avg_test_loss = sum(local_test_losses) / len(local_test_losses)
        local_avg_test_losses_list.append(loss_avg_test_loss)
        loss_avg_test_accuracy = sum(local_test_accuracy) / len(local_test_accuracy)
        local_avg_test_accuracy_list.append(loss_avg_test_accuracy)

        if (epoch+1) % print_every == 0:
            print(f' \nAvg Stats after {epoch+1} global rounds:')
            print(f'Local Avg Training Loss : {loss_avg}')
            print(f'Local Avg Test Loss : {loss_avg_test_loss}')
            print(f'Local Avg Test Accuracy : {loss_avg_test_accuracy}')
        previous_user_list = idxs_users

        local_model_dict2 = copy.deepcopy(current_user_id_modelweights_dict)
        local_model_dict = copy.deepcopy(selected_models)

        client_model_info = client_model_dict


        server_output_dict = layer_feature(local_model_dict, server_train_data, client_model_info, args)
        
        model_layer_index_to_model_layer_name = process_format(server_output_dict)

        prepare_layer_size_dict = {}
        for key, value in server_output_dict.items():
            prepare_layer_size_dict[str(model_layer_index_to_model_layer_name[key])] = value[-2:]


        server_output_dict_only_embedding = extract_embedding(server_output_dict)
        server_output_dict_only_size = extrac_size(server_output_dict)

        server_output_dict_same_embedding = embedding_process(server_output_dict)
        

        cluster_results = k_cluster(server_output_dict_same_embedding, args.cluster_num, -5, 10)

        for_find_comb_input = {}
        for key, value in cluster_results.items():
            new_value = []
            for item in value:
                temp_temp_temp = model_layer_index_to_model_layer_name[item]
                new_value.append(temp_temp_temp)
            for_find_comb_input[key] = new_value
    


        candidate_model_combine = sample_models(for_find_comb_input, 4, 6, args.expected_num_models)

        print(candidate_model_combine)
        model_pool_with_mlp = comb_with_mlp(candidate_model_combine, local_model_dict, prepare_layer_size_dict, input_size = input_size)

        if args.supervised:
            best_model_dict = match_best_model(model_pool_with_mlp,local_model_dict2,server_train_data)
        else:
            best_model_dict = match_best_model_2(model_pool_with_mlp,local_model_dict2,server_train_data) # list is the candidate model pool, local_model_dict is previous client model dict,
   

    save_path = './exp_result/{}_{}_com{}_iid_{}_E_{}_sfine_{}_teacherweight_{}_cluster_{}_frac_{}_user_num_{}/'.format(args.dataset, args.public_dataset, args.epochs,
                       args.iid, args.local_ep, args.sfine, args.alpha, args.cluster_num, args.frac, args.num_users)

    isExist = os.path.exists(save_path)

    if not isExist:
        os.makedirs(save_path)
        print("The new directory is created!")
    


        
        












