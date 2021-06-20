import pickle
import numpy as np
import random, string, os, sys, io
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import basinhopping
from sklearn.metrics import f1_score
from timeit import default_timer as timer
from argparse import ArgumentParser
import copy
from custom_metrics import hamming_score, f1
from custom_dataset import batch_maker, flattern_result

from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.base import scope
import torch
random.seed(42)

tune_best_model = []
max_utterance_num = 0
max_user_history_num = 90

# from transformers import BertTokenizer
# pretrained_weights = './bert/'
# tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def best_score_search(true_labels, predictions, f):
    def f_neg(threshold):
        ## Scipy tries to minimize the function so we must get its inverse
        return - f(true_labels, pd.DataFrame(predictions).values > pd.DataFrame(threshold).values.reshape(1, len(predictions[0])))

    thr_0 = [0.20] * len(predictions[0])
    constraints = [(0.,1.)] * len(predictions[0])
    def bounds(**kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= 1))
        tmin = bool(np.all(x >= 0)) 
        return tmax and tmin

    # Search using L-BFGS-B, the epsilon step must be big otherwise there is no gradient
    minimizer_kwargs = {"method": "L-BFGS-B",
                        "bounds":constraints,
                        "options":{
                            "eps": 0.05
                            }
                       }
    
    # We combine L-BFGS-B with Basinhopping for stochastic search with random steps
    print("===> Searching optimal threshold for each label")
    start_time = timer()

    opt_output = basinhopping(f_neg, thr_0,
                                stepsize = 0.1,
                                minimizer_kwargs=minimizer_kwargs,
                                niter=10,
                                accept_test=bounds)

    end_time = timer()
    print("===> Optimal threshold for each label:\n{}".format(opt_output.x))
    print("Threshold found in: %s seconds" % (end_time - start_time))

    score = - opt_output.fun
    return score, opt_output.x

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def ret_index (li, s):
    if s in li:
        return li.index(s)
    else:
        # print(s)
        return -1

def str2vector(li, text, tag, max_utterance_lengths):
    if tag=="text":
        max_len = max_utterance_lengths 
        ret = [ li[s]+1 for s in text.split()]
        ret += [0] * (max_len - len(ret))

        # max_len = max_utterance_lengths
        # ret = tokenizer.encode(text, add_special_tokens=False)
        # if len(ret) > max_len-2:
        #     _l = ret[:(max_len-2)//2]
        #     _r = ret[(max_len-2)//2-(max_len-2):]
        #     ret = _l + _r
        # ret = [101] + ret + [102]  # 101[CLS]  102[SEP]  0[PAD]
        # ret += [0] * (max_len - len(ret))

        # print(text)
    elif tag=="da":
        if len(text) == 0:
            return [0] * len(li)
        count = [ ret_index(li, s) for s in text.split()]
        ret = [0] * len(li)
        for c in count:
            assert c >= 0
            ret[c] = 1
    else:
        count = ret_index(li, text)
        assert count >= 0
        ret = [count]
    return ret

def userlist2graph_inside(user_list):
    u_num = len(user_list)
    edge_x = []
    edge_y = []
    edge_z = []
    for i_n in range(u_num):
        for j_n in range(i_n):
            edge_x.append(i_n)
            edge_y.append(j_n)
            if user_list[i_n] == user_list[j_n]:
                edge_z.append(2)
            else:
                edge_z.append(3)
        for j_n in range(i_n+1, u_num):
            edge_x.append(i_n)
            edge_y.append(j_n)
            if user_list[i_n] == user_list[j_n]:
                edge_z.append(0)
            else:
                edge_z.append(1)
    return [edge_x, edge_y, edge_z]

def userlist2graph_ext(user_list, num_classes):
    u_num = len(user_list)
    edge_x = []
    edge_y = []
    edge_z = []
    for i_n in range(u_num):
        for c_n in range(num_classes):
            c_idx = u_num + c_n
            edge_x.append(c_idx)
            edge_y.append(i_n)
            edge_z.append(4)
            #edge_index.append([i_n, c_idx, 5])
    """
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                edge_index.append([i, j, 6])
    """
    return [edge_x, edge_y, edge_z]

def userlist2graph(user_list, num_classes):
    graph_inside = userlist2graph_inside(user_list)
    graph_ext = userlist2graph_ext(user_list, num_classes)
    edge_x = graph_inside[0] + graph_ext[0]
    edge_y = graph_inside[1] + graph_ext[1]
    edge_z = graph_inside[2] + graph_ext[2]
    return [edge_x, edge_y, edge_z]

def userlist2tag(user_list):
    n_num = len(user_list)
    u_tag = np.zeros((2, n_num, n_num))
    u_set = []
    for i_n in range(n_num):
        nam = user_list[i_n]
        if nam not in u_set:
            u_idx = len(u_set)
            u_tag[0][u_idx][i_n] = 1
            u_set.append(nam)
    u_set_reverse = []
    for i_n in range(n_num-1, -1, -1):
        nam = user_list[i_n]
        if nam not in u_set_reverse:
            u_idx = u_set.index(nam)
            u_tag[1][u_idx][i_n] = 1
            u_set_reverse.append(nam)
    return u_tag.tolist()

def user2history(user, history_dict):
    if not history_dict or history_dict.get(user, None) is None:
        return None
    ret = history_dict[user]
    return ret

def userlist2history(conv_no, conv_cls_his_dict, max_user_history_num=100):
    his_list = copy.deepcopy(conv_cls_his_dict[conv_no])
    num_classes = len(his_list)
    for i in range(num_classes):
        his_l = his_list[i]
        random.shuffle(his_l)
        his_l = his_l[:max_user_history_num]
        his_list[i] = his_l + ([0] * (max_user_history_num-len(his_l)))
    return his_list

def user2hismask(user, history_dict, num_classes=12):
    if not history_dict or history_dict.get(user, None) is None:
        return None
    his = history_dict[user]
    ret = []
    for i in range(num_classes):
        ret.append(len(his[i]))
    return ret

def userlist2hismask(user_history):
    num_cls = len(user_history)
    ret = []
    for i in range(num_cls):
        his = user_history[i]
        msk = []
        for j in range(len(his)):
            if his[j] == 0:
                msk.append(0)
            else:
                msk.append(1)
        ret.append(msk)
    return ret

def ret_predict(predicts, thresholds, discount=1.0):
    thresholds = [t * discount for t in thresholds]
    ret = [int(val >= thresholds[idx]) for idx, val in enumerate(predicts)]
    if sum(ret) == 0:
        if discount < 1.0:
            ret = ret_predict(predicts, thresholds, discount)
    return ret

def main(args):

    # data loading
    dim = args.dim
    seed = args.random
    max_length = args.max_len
    print()
    print("random seed", seed)
    print("word embedding dimension", dim)

    if sys.argv[1] == 'tune':
        tuning = True
    else:
        tuning = False

    # read in curpus
    ms_tags = ['CQ', 'FD', 'FQ', 'GG', 'IR', 'JK', 'NF', 'O', 'OQ', 'PA', 'PF', 'RQ']

    #ms_tags = ["s", "qy", "qw", "qr", "qrr", "qo", "qh", "b", "fg", "fh", "h", 
    #"aa", "aap", "am", "ar", "arp", "ba", "bc", "bd", "bh", "bk", 
    #"br", "bs", "bsc", "bu", "by", "cc", "co", "cs", "d", "df", 
    #"e", "f", "fa", "fe", "ft", "g", "j", "m", "na", "nd", 
    #"ng", "no", "r", "rt", "t", "tc", "t1", "t3", "2"]#, "fw"]

    ms_entitiedbowpath = os.path.normpath("./data/msdialog.csv")

    df = pd.read_csv(ms_entitiedbowpath)

    # conversation_numbers = df['conversation_no']
    utterance_tags = df['tags']
    utterances = df['utterance']
    utterance_status = df['utterance_status']
    actor_names = df['user_id']
    utterance_id = df['utterance_id']

    max_utterance_lengths = max_length
    global max_utterance_num
    max_utterance_num = len(utterances)
    print('max utterance length', max_utterance_lengths)

    all_dialogs = []
    all_tags = []
    all_user_names = []
    all_utt_ids = []
    all_conv_no = []

    for i in range(len(utterances)):
        if utterance_status[i] == "B":
            dialog_utterances = [' '.join(utterances[i].split()[:max_length])]
            dialog_tags = [utterance_tags[i]]
            user_names = [actor_names[i]]
            utt_id = [utterance_id[i]]
        else:
            dialog_utterances.append(' '.join(utterances[i].split()[:max_length]))
            dialog_tags.append(utterance_tags[i])
            user_names.append(actor_names[i])
            utt_id.append(utterance_id[i])
            if utterance_status[i] == 'E':
                all_dialogs.append(dialog_utterances)
                all_tags.append(dialog_tags)
                all_user_names.append(user_names)
                all_utt_ids.append(utt_id)
                all_conv_no.append(len(all_conv_no))

    X_train, X_val, y_train, y_val, z_train, z_val, n_train, n_val, p_train, p_val = train_test_split(all_dialogs, all_tags, all_user_names, all_utt_ids, all_conv_no, test_size=0.1, shuffle=False)
    X_train, X_test, y_train, y_test, z_train, z_test, n_train, n_test, p_train, p_test = train_test_split(X_train, y_train, z_train, n_train, p_train, test_size=0.1, shuffle=False)

    #X_train, X_val, y_train, y_val, z_train, z_val, n_train, n_val, p_train, p_val = train_test_split(all_dialogs, all_tags, all_user_names, all_utt_ids, all_conv_no, test_size=989.0/3429.0, shuffle=False)
    #X_val, X_test, y_val, y_test, z_val, z_test, n_val, n_test, p_val, p_test = train_test_split(X_val, y_val, z_val, n_val, p_val, test_size=488.0/989.0, shuffle=False)

    counts_train = [len(x) for x in y_train]
    counts_test = [len(x) for x in y_test]
    counts_val = [len(x) for x in y_val]

    print('Statistics of training set:')
    print('Utterances:', sum(counts_train))
    print('Min. # Turns Per Dialog', min(counts_train))
    print('Max. # Turns Per Dialog', max(counts_train))
    print('Avg. # Turns Per Dialog:', sum(counts_train)/len(counts_train))
    print('Avg. # DAs Per Utterance', sum(sum(len(y.split()) for y in x) for x in y_train)/sum(counts_train))
    print('Avg. # Words Per Utterance', sum(sum(len(y.split()) for y in x) for x in X_train)/sum(counts_train))
    print()
    print('Statistics of validation set:')
    print('Utterances:', sum(counts_val))
    print('Min. # Turns Per Dialog', min(counts_val))
    print('Max. # Turns Per Dialog', max(counts_val))
    print('Avg. # Turns Per Dialog:', sum(counts_val)/len(counts_val))
    print('Avg. # DAs Per Utterance', sum(sum(len(y.split()) for y in x) for x in y_val)/sum(counts_val))
    print('Avg. # Words Per Utterance', sum(sum(len(y.split()) for y in x) for x in X_val)/sum(counts_val))
    print()
    print('Statistics testing sets:')
    print('Utterances:', sum(counts_test))
    print('Min. # Turns Per Dialog', min(counts_test))
    print('Max. # Turns Per Dialog', max(counts_test))
    print('Avg. # Turns Per Dialog:', sum(counts_test)/len(counts_test))
    print('Avg. # DAs Per Utterance', sum(sum(len(y.split()) for y in x) for x in y_test)/sum(counts_test))
    print('Avg. # Words Per Utterance', sum(sum(len(y.split()) for y in x) for x in X_test)/sum(counts_test))


    # read in dict
    target_vocab = []
    with open('./data/msdialog_bow.tab', 'r', encoding='utf8') as f:
        for line in f:
            items = line.split('\t')
            key, value = items[0], int(items[1])
            target_vocab.append(key)

    output_size = len(ms_tags)
    word_to_ix = {word: i for i, word in enumerate(target_vocab)}

    glove = {}
    with open("./data/glove.6B.100d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            itmes = line.strip().split()
            glove[itmes[0]] = list(map(float, itmes[1:]))

    # with padding vector
    matrix_len = len(target_vocab) + 1
    weights_matrix = np.zeros((matrix_len, dim))

    # the padding vector
    weights_matrix[0] = np.zeros((dim, ))

    words_found = 0

    for i, word in enumerate(target_vocab):
        try: 
            weights_matrix[i+1] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i+1] = np.random.normal(scale=0.6, size=(dim, ))

    print(words_found,'/',len(target_vocab), 'words found embeddings')

    utt_weight_matrix = None
    if args.usergraph:
        utt_weight_matrix = pickle.load(open(args.model[0]+"user"+str(args.model_file), "rb"))
        utt_weight_matrix = torch.Tensor(utt_weight_matrix)

    conv_cls_his = pickle.load(open("./data/msdialog_conv_his.pkl", "rb"))

    print('Preparing data ...')

    train_n_iters = len(X_train)
    train_data = [ [str2vector(word_to_ix, sent, "text", max_utterance_lengths) for sent in X_train[i]] for i in range(train_n_iters)]
    train_target = [ [str2vector(ms_tags, sent, "da", max_utterance_lengths) for sent in y_train[i]] for i in range(train_n_iters)]
    train_user_names = z_train
    train_utt_id = n_train
    train_user_tree = [userlist2graph(z_train[i], output_size) for i in range(train_n_iters)]
    train_user_tag = [userlist2tag(z_train[i]) for i in range(train_n_iters)]
    train_user_history = [userlist2history(p_train[i], conv_cls_his, max_user_history_num) for i in range(train_n_iters)]
    train_user_hismask = [userlist2hismask(train_user_history[i]) for i in range(train_n_iters)]

    val_n_iters = len(X_val)

    val_data = [ [str2vector(word_to_ix, sent, "text", max_utterance_lengths) for sent in X_val[i]] for i in range(val_n_iters)]
    val_target = [ [str2vector(ms_tags, sent, "da", max_utterance_lengths) for sent in y_val[i]] for i in range(val_n_iters)]
    val_user_names = z_val
    val_utt_id = n_val
    val_user_tree = [userlist2graph(z_val[i], output_size) for i in range(val_n_iters)]
    val_user_tag = [userlist2tag(z_val[i]) for i in range(val_n_iters)]
    val_user_history = [userlist2history(p_val[i], conv_cls_his, max_user_history_num) for i in range(val_n_iters)]
    val_user_hismask = [userlist2hismask(val_user_history[i]) for i in range(val_n_iters)]

    test_n_iters = len(X_test)

    test_data = [ [str2vector(word_to_ix, sent, "text", max_utterance_lengths) for sent in X_test[i]] for i in range(test_n_iters)]
    test_target = [ [str2vector(ms_tags, sent, "da", max_utterance_lengths) for sent in y_test[i]] for i in range(test_n_iters)]
    test_user_names = z_test
    test_utt_id = n_test
    test_user_tree = [userlist2graph(z_test[i], output_size) for i in range(test_n_iters)]
    test_user_tag = [userlist2tag(z_test[i]) for i in range(test_n_iters)]
    test_user_history = [userlist2history(p_test[i], conv_cls_his, max_user_history_num) for i in range(test_n_iters)]
    test_user_hismask = [userlist2hismask(test_user_history[i]) for i in range(test_n_iters)]

    if not tuning:
        run(args, weights_matrix, output_size, utt_weight_matrix,
            train_data, train_target, train_user_names, train_utt_id, train_user_tree, train_user_tag, train_user_history, train_user_hismask,
            val_data, val_target, val_user_names, val_utt_id, val_user_tree, val_user_tag, val_user_history, val_user_hismask,
            test_data, test_target, test_user_names, test_utt_id, test_user_tree, test_user_tag, test_user_history, test_user_hismask, tuning, ms_tags)
    else:
        def objective(params):
            if 'lstm_hidden' in params.keys():
                args.lstm_hidden = params['lstm_hidden']
            if 'lr' in params.keys():
                args.lr = params['lr']
            if 'filters' in params.keys():
                args.filters = params['filters']
            if 'cd' in params.keys():
                args.cd = params['cd']
            if 'ld' in params.keys():
                args.ld = params['ld']
            # args.max_len = params['max_len']
            return run(args, weights_matrix, output_size, utt_weight_matrix,
                        train_data, train_target, train_user_names, train_utt_id, train_user_tree, train_user_tag, train_user_history, train_user_hismask,
                        val_data, val_target, val_user_names, val_utt_id, val_user_tree, val_user_tag, val_user_history, val_user_hismask,
                        test_data, test_target, test_user_names, test_utt_id, test_user_tree, test_user_tag, test_user_history, test_user_hismask, tuning, ms_tags)

        space = {
            'lstm_hidden': scope.int(hp.quniform('lstm_hidden', 300, 600, 100)),
            'lr': hp.quniform('lr', 1e-5, 5e-5, 1e-5),
            'filters': scope.int(hp.quniform('filters', 100, 200, 50)),
            'cd': hp.quniform('cd', 0.2, 0.4, 0.05),
            'ld': hp.quniform('ld', 0.1, 0.5, 0.1),
        }

        best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)

        best_params = space_eval(space,best_params)

        print(best_params)

        tune_acc_best = sorted(tune_best_model, key=lambda x:x[0], reverse=True)
        tune_f1_best = sorted(tune_best_model, key=lambda x:x[3], reverse=True)
        with io.open("acc.log", 'w', encoding="utf-8") as f:
            for item in tune_acc_best:
                f.write(str(item))
                f.write('\n')
        with io.open("f1.log", 'w', encoding="utf-8") as f:
            for item in tune_f1_best:
                f.write(str(item))
                f.write('\n')
        return best_params


def run(args, weights_matrix, output_size, utt_weight_matrix,
        train_data, train_target, train_user_names, train_utt_id, train_user_tree, train_user_tag, train_user_history, train_user_hismask,
        val_data, val_target, val_user_names, val_utt_id, val_user_tree, val_user_tag, val_user_history, val_user_hismask,
        test_data, test_target, test_user_names, test_utt_id, test_user_tree, test_user_tag, test_user_history, test_user_hismask, tuning, ms_tags):
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    from torch.autograd import Variable
    from torch import optim
    from HUH import HUH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_matrix = torch.Tensor(weights_matrix)
    if utt_weight_matrix is not None:
        utt_weight_matrix = torch.Tensor(utt_weight_matrix)

    if tuning or sys.argv[1] == 'train':

        # Global setup
        hidden_size = args.lstm_hidden
        n_epochs = args.epoch
        criterion = nn.BCELoss()
        patient = args.patient
        learning_rate = args.lr
        n_filters = args.filters
        filter_sizes = args.filter_sizes
        c_dropout = args.cd
        l_dropout = args.ld
        batch_size = args.batch_size

        save_path = './model/'+randomword(10)+'/'
        while os.path.exists(save_path):
            save_path = './model/'+randomword(10)+'/'

        if not tuning: 
            print()
            print('Parameters')
            print('lstm_hidden_size', hidden_size)
            print('epochs', n_epochs)
            print('patient', patient)
            print('learning_rate', learning_rate)
            print('n_filters', n_filters)
            print('filter_sizes', filter_sizes)
            print('batch_size', batch_size)
            print('CNN dropout', c_dropout)
            print('LSTM dropout', l_dropout)
            print()
        print('model will be saved to', save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        args_file = open(save_path+"args", "w", encoding="utf-8")
        args_file.write(str(args))
        args_file.close()

        # torch.backends.cudnn.enabled = False
        model = HUH(hidden_size, output_size, weights_matrix, n_filters, filter_sizes, c_dropout, l_dropout, utt_weight_matrix, max_utterance_num)
        model = model.to(device)

        if args.usergraph:
            model.load_state_dict(torch.load(args.model[0]+str(args.model_file)))
            model.utt_embedding.load_state_dict({'weight': utt_weight_matrix})
            model.utt_embedding.weight.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            frozen_layers = [model.edge_att, model.graph_net, model.utt_embedding, model.sim_w]
            for layer in frozen_layers:
                for k, v in layer.named_parameters():
                    v.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

        losses = np.zeros(n_epochs)
        vlosses = np.zeros(n_epochs)

        best_epoch = 0
        stop_counter = 0
        best_score = None

        train_loader_dataset = batch_maker(train_data, train_target, train_user_names, train_utt_id, train_user_tree, train_user_tag, train_user_history, train_user_hismask, batch_size)
        val_loader_dataset = batch_maker(val_data, val_target, val_user_names, val_utt_id, val_user_tree, val_user_tag, val_user_history, val_user_hismask, batch_size)
        # learning
        for epoch in range(n_epochs):
            ###################
            # train the model #
            ###################
            model.train() # prep model for training

            if not args.usergraph:
                t_utt_weight_matrix = np.zeros((max_utterance_num+1, hidden_size * 2))
                t_utt_weight_matrix[0] = np.zeros((hidden_size * 2,))

            for data in train_loader_dataset:
                src_seqs, trg_seqs, usr_names, utt_ids, usr_tree, usr_tag, usr_his, usr_hismask = data
                src_seqs, trg_seqs, usr_tag, usr_his, usr_hismask = src_seqs.to(device), trg_seqs.to(device), usr_tag.to(device), usr_his.to(device), usr_hismask.to(device)

                cxt, outputs, graph_outputs = model(src_seqs, usr_tree, usr_tag, usr_his, usr_hismask, trg_seqs)

                if not args.usergraph:
                    cxt = cxt.detach().cpu().numpy()
                    for i in range(len(utt_ids)):
                        for j in range(len(utt_ids[i])):
                            t_utt_weight_matrix[utt_ids[i][j]] = cxt[i][j]

                # print(outputs)
                optimizer.zero_grad()
                if args.usergraph:
                    graph_outputs = graph_outputs.to(device)
                    loss = criterion(graph_outputs, trg_seqs)
                else:
                    outputs = outputs.to(device)
                    loss = criterion(outputs, trg_seqs)
                loss.backward()
                optimizer.step()
                # print(loss.item())
                losses[epoch] += loss.item()
            if not tuning:
                print('epoch', epoch+1, ' average train loss: ', losses[epoch] / len(train_loader_dataset))

            torch.cuda.empty_cache()

            ######################    
            # validate the model #
            ######################
            model.eval() # prep model for evaluation

            for i, data in enumerate(val_loader_dataset, 0):
                src_seqs, trg_seqs, usr_names, utt_ids, usr_tree, usr_tag, usr_his, usr_hismask = data
                src_seqs, trg_seqs, usr_tag, usr_his, usr_hismask = src_seqs.to(device), trg_seqs.to(device), usr_tag.to(device), usr_his.to(device), usr_hismask.to(device)

                cxt, outputs, graph_outputs = model(src_seqs, usr_tree, usr_tag, usr_his, usr_hismask)

                if args.usergraph:
                    graph_outputs = graph_outputs.to(device)
                    loss = criterion(graph_outputs, trg_seqs).item()
                else:
                    outputs = outputs.to(device)
                    loss = criterion(outputs, trg_seqs).item()
                vlosses[epoch] += loss
            if not tuning:
                print('epoch', epoch+1, ' average val loss: ', vlosses[epoch] / len(val_loader_dataset))

            if best_score is None or vlosses[epoch] < best_score:
                best_score = vlosses[epoch]
                best_epoch = epoch+1
                torch.save(model.state_dict(), save_path+str(best_epoch))
                if args.usergraph:
                    t_utt_weight_matrix = utt_weight_matrix
                pickle.dump(t_utt_weight_matrix, open(save_path+"user"+str(best_epoch), "wb"))
                stop_counter = 0
                if not tuning:
                    print('epoch', best_epoch, 'model updated')
            else:
                stop_counter += 1

            torch.cuda.empty_cache()

            if stop_counter >= patient:
                print("Early stopping")
                break
        if not tuning:
            print('Models saved to', save_path)
            print('Best epoch', str(best_epoch), ', with score', str(best_score / len(val_loader_dataset)))

    if tuning or (sys.argv[1] == 'test' and len(sys.argv) > 2 and sys.argv[1] != ''):

        criterion = nn.BCELoss()
        test_discount = 1.0

        if tuning:
            directory = save_path
            epoch = best_epoch
        else:
            directory = args.model[0]
            epoch = args.model_file

            # Global setup
            hidden_size = args.lstm_hidden
            n_filters = args.filters
            filter_sizes = args.filter_sizes
            c_dropout = args.cd
            l_dropout = args.ld
            test_discount = args.discount
            batch_size = args.batch_size

        if not tuning:
            print('lstm_hidden_size', hidden_size)
            print('n_filters', n_filters)
            print('filter_sizes', filter_sizes)
            print('batch_size', batch_size)
            print('CNN dropout', c_dropout)
            print('LSTM dropout', l_dropout)
            print('test discount', test_discount)

        bloss = 9999999.99;
        breferences = []
        bpredicts = []
        bfile = ''

        model = HUH(hidden_size, output_size, weights_matrix, n_filters, filter_sizes, c_dropout, l_dropout, utt_weight_matrix, max_utterance_num)
        model = model.to(device)

        for filename in os.listdir(directory):
            if '.' in filename: continue
            if epoch > 0 and filename != str(epoch):
                continue

            model.load_state_dict(torch.load(directory+filename))
            model.eval()

            train_loader_dataset = batch_maker(train_data, train_target, train_user_names, train_utt_id, train_user_tree, train_user_tag, train_user_history, train_user_hismask, batch_size)
            val_loader_dataset = batch_maker(val_data, val_target, val_user_names, val_utt_id, val_user_tree, val_user_tag, val_user_history, val_user_hismask, batch_size)
            test_loader_dataset = batch_maker(test_data, test_target, test_user_names, test_utt_id, test_user_tree, test_user_tag, test_user_history, test_user_hismask, batch_size)

            loss = 0.0
            references = None
            predicts = None
            for data in val_loader_dataset:
                src_seqs, trg_seqs, usr_names, utt_ids, usr_tree, usr_tag, usr_his, usr_hismask = data
                src_seqs, trg_seqs, usr_tag, usr_his, usr_hismask = src_seqs.to(device), trg_seqs.to(device), usr_tag.to(device), usr_his.to(device), usr_hismask.to(device)

                cxt, outputs, graph_outputs = model(src_seqs, usr_tree, usr_tag, usr_his, usr_hismask)

                # print(outputs)
                if args.usergraph:
                    graph_outputs = graph_outputs.to(device)
                    loss += criterion(graph_outputs, trg_seqs).item()
                else:
                    outputs = outputs.to(device)
                    loss += criterion(outputs, trg_seqs).item()

                reference = flattern_result(trg_seqs.cpu().numpy())
                if args.usergraph:
                    predict = flattern_result(graph_outputs.detach().cpu().numpy())
                else:
                    predict = flattern_result(outputs.detach().cpu().numpy())

                if references is None or predicts is None:
                    references = reference
                    predicts = predict
                else:
                    # print(predicts, predict)
                    references = np.append(references, reference, axis=0)
                    predicts = np.append(predicts, predict, axis=0)

            vloss = loss / len(val_loader_dataset)
            if not tuning:
                print('Epoch', filename, 'average val loss: ', vloss)

            if vloss < bloss:
                bloss = vloss
                breferences = np.array(references)
                bpredicts = np.array(predicts)
                bfile = filename

            torch.cuda.empty_cache()

        best_score, thresholds = best_score_search(breferences, bpredicts, hamming_score)
        if not tuning:
            print('best validation epoch:', bfile, 'with score:', str(best_score))

        # load the best model
        model.load_state_dict(torch.load(directory+bfile))
        model.eval()

        loss = 0.0 # For plotting
        references = None
        predicts = None

        for data in test_loader_dataset:
            src_seqs, trg_seqs, usr_names, utt_ids, usr_tree, usr_tag, usr_his, usr_hismask = data
            src_seqs, trg_seqs, usr_tag, usr_his, usr_hismask = src_seqs.to(device), trg_seqs.to(device), usr_tag.to(device), usr_his.to(device), usr_hismask.to(device)

            cxt, outputs, graph_outputs = model(src_seqs, usr_tree, usr_tag, usr_his, usr_hismask)

            # print(outputs)
            if args.usergraph:
                graph_outputs = graph_outputs.to(device)
                loss += criterion(graph_outputs, trg_seqs).item()
            else:
                outputs = outputs.to(device)
                loss += criterion(outputs, trg_seqs).item()

            reference = flattern_result(trg_seqs.cpu().numpy())
            if args.usergraph:
                predict = flattern_result(graph_outputs.detach().cpu().numpy())
            else:
                predict = flattern_result(outputs.detach().cpu().numpy())

            if references is None or predicts is None:
                references = reference
                predicts = predict
            else:
                references = np.append(references, reference, axis=0)
                predicts = np.append(predicts, predict, axis=0)

        tloss = loss / len(test_loader_dataset)
        if not tuning:
            print('average test loss: ', tloss)

        torch.cuda.empty_cache()

        predictions = []

        for j in range(len(predicts)):
            predictions.append(ret_predict(predicts[j], thresholds, discount=args.discount))

        references = np.array(references);
        predictions = np.array(predictions);

        acc = hamming_score(y_true=references, y_pred=predictions)
        f1_scores = f1(y_true=references, y_pred=predictions)

        scores = str(acc) + ',' + ','.join([str(x) for x in f1_scores])
        print('Test Accuracy, Precision, Recall and F1 score: ', scores)

        if tuning:
            tun_model = [acc]
            tun_model.extend(f1_scores)
            tun_model.append(directory)
            tune_best_model.append(tun_model)

        if not tuning:
            print('Tag',':','Accuracy, (Precision, Recall, F1)')
            for i in range(predictions.shape[1]):
                predictions_t = np.array([[p[i]] for p in predictions])
                references_t = np.array([[r[i]]for r in references])
                print(ms_tags[i], ':',hamming_score(y_true=references_t, y_pred=predictions_t),',', f1(y_true=references_t, y_pred=predictions_t))

        return {'loss': -acc, 'status': STATUS_OK }

if __name__ == "__main__":
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(help='commands')

    # A train command
    train_parser = subparsers.add_parser('train', help='train the model')
    train_parser.add_argument('--model', type=str, nargs=1, help='directory for model files')
    train_parser.add_argument('--model_file', type=int, default=0, nargs='?', help='specify the epoch to test')
    train_parser.add_argument('--lstm_hidden', type=int, default=200, nargs='?', help='hidden size in output MLP')
    train_parser.add_argument('--dim', type=int, default=100, nargs='?', help='dimension of word embeddings')
    train_parser.add_argument('--epoch', type=int, default=1000, nargs='?', help='number of epochs to run')
    train_parser.add_argument('--patient', type=int, default=2, nargs='?', help='number of epochs to wait if no improvement and then stop the training.')
    train_parser.add_argument('--lr', type=float, default=0.001, nargs='?', help='learning rate')
    train_parser.add_argument('--filter_sizes', type=int, default=[3,4,5], nargs='+', help='filter sizes')
    train_parser.add_argument('--filters', type=int, default=100, nargs='?', help='number of CNN kernel filters.')
    train_parser.add_argument('--random', type=int, default=42, nargs='?', help='random seed')
    train_parser.add_argument('--cd', type=float, default=0.5, nargs='?', help='CNN dropout')
    train_parser.add_argument('--ld', type=float, default=0.5, nargs='?', help='LSTM dropout')
    train_parser.add_argument('--max_len', type=int, default=80, nargs='?', help='max length of utterance')
    train_parser.add_argument('--batch_size', type=int, default=1, nargs='?', help='batch size')
    train_parser.add_argument("--usergraph", type=str2bool, nargs='?',const=True, default=False, help="User Intent task")
    train_parser.add_argument('--discount', type=float, default=1, nargs='?', help='test discount')

    # A tuning command
    tune_parser = subparsers.add_parser('tune', help='tune the model')
    tune_parser.add_argument('--model', type=str, nargs=1, help='directory for model files')
    tune_parser.add_argument('--model_file', type=int, default=0, nargs='?', help='specify the epoch to test')
    tune_parser.add_argument('--lstm_hidden', type=int, default=200, nargs='?', help='hidden size in output MLP')
    tune_parser.add_argument('--dim', type=int, default=100, nargs='?', help='dimension of word embeddings')
    tune_parser.add_argument('--epoch', type=int, default=1000, nargs='?', help='number of epochs to run')
    tune_parser.add_argument('--patient', type=int, default=2, nargs='?', help='number of epochs to wait if no improvement and then stop the training.')
    tune_parser.add_argument('--filter_sizes', type=int, default=[3,4,5], nargs='+', help='filter sizes')
    tune_parser.add_argument('--filters', type=int, default=100, nargs='?', help='number of CNN kernel filters.')
    tune_parser.add_argument('--random', type=int, default=42, nargs='?', help='random seed')
    tune_parser.add_argument('--cd', type=float, default=0.5, nargs='?', help='CNN dropout')
    tune_parser.add_argument('--ld', type=float, default=0.05, nargs='?', help='LSTM dropout')
    tune_parser.add_argument('--max_len', type=int, default=80, nargs='?', help='max length of utterance')
    tune_parser.add_argument('--batch_size', type=int, default=1, nargs='?', help='batch size')
    tune_parser.add_argument("--usergraph", type=str2bool, nargs='?',const=True, default=False, help="User Intent task")
    tune_parser.add_argument('--discount', type=float, default=1, nargs='?', help='test discount')

     # A test command
    test_parser = subparsers.add_parser('test', help='test the model')
    test_parser.add_argument('--model', type=str, nargs=1, help='directory for model files', required=True)
    test_parser.add_argument('--model_file', type=int, default=0, nargs='?', help='specify the epoch to test')
    test_parser.add_argument('--lstm_hidden', type=int, default=200, nargs='?', help='hidden size in output MLP')
    test_parser.add_argument('--dim', type=int, default=100, nargs='?', help='dimension of word embeddings')
    test_parser.add_argument('--filters', type=int, default=100, nargs='?', help='number of CNN kernel filters.')
    test_parser.add_argument('--filter_sizes', type=int, default=[3,4,5], nargs='+', help='filter sizes')
    test_parser.add_argument('--random', type=int, default=42, nargs='?', help='random seed')
    test_parser.add_argument('--cd', type=float, default=0.5, nargs='?', help='CNN dropout')
    test_parser.add_argument('--ld', type=float, default=0.05, nargs='?', help='LSTM dropout')
    test_parser.add_argument('--max_len', type=int, default=80, nargs='?', help='max length of utterance')
    test_parser.add_argument('--batch_size', type=int, default=1, nargs='?', help='batch size')
    test_parser.add_argument("--usergraph", type=str2bool, nargs='?',const=True, default=False, help="User Intent task")
    test_parser.add_argument('--discount', type=float, default=1, nargs='?', help='test discount')

    dataset_parser = subparsers.add_parser('dataset', help='save the dataset files')

    args = parser.parse_args()
    main(args)
