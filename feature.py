import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from load_datas import OurData, Dataloader

device = "cuda:2"

# def de_dim(train_r_data, train_w_data, valid_r_data, valid_w_data, test_r_data, test_w_data):
#     S = torch.cat([train_r_data['s2'], valid_r_data['s2'], test_r_data['s2']], dim=0)
#     print(S.size())
#     for i in range(S.size(1)):
#         S[:,i] = (S[:,i] - S[:,i].min()) / S[:,i].max()
    
#     n = S.size(0)
#     S_centered = S - torch.mean(S, dim=0)
#     del train_r_data, train_w_data, valid_r_data, valid_w_data, test_r_data, test_w_data
#     del S

#     S_centered = S_centered.to(device)
#     cov_matrix = torch.mm(S_centered.t(), S_centered) / (n - 1)
#     e_values, e_vectors = torch.linalg.eigh(cov_matrix)

#     sorted_indices = torch.argsort(e_values, descending=True)
#     e_vectors = e_vectors[:, sorted_indices]

#     p_comp = e_vectors[:, :128]
#     S_reduced = torch.mm(S_centered, p_comp)
#     print(S_reduced.size())
#     torch.save(S_reduced, "low_dim_state.pt")

def get_features(data: OurData, print_size=False):
    """
    train/valid/test:
        _r_data:
            {
            "s1": None, 为上一条数据的s2 (为了省内存)
            "a": list[int], worker的id, 大小为N
            "r": list[float], 得分，大小为N
            "s2": tensor (N, dim)
            }
        _w_data:
            {
            "s1": None, 为上一条数据的s2 (为了省内存)
            "a": list[int], 是否接了任务, 大小为N
            "r": list[float], 钱，大小为N
            "s2": tensor (N, dim)
            }
    """
    ws, ps, es = data.workers, data.projects, data.entrys
    requester_data, worker_data = {"s1": [], "a": [], "r": [], "s2": []}, {"s1": [], "a": [], "r": [], "s2": []}

    wws = {}
    for k, v in ws.items():
        if v != -1:
            wws[int(k)] = v

    w_num = len(wws)
    p_num = len(ps)
    e_num = len(es)
    w_id2index, w_index2id = {}, list(wws.keys())
    for idx, _id in enumerate(w_index2id):
        w_id2index[_id] = idx
    p_id2index, p_index2id = {}, list(ps.keys())
    for idx, _id in enumerate(p_index2id):
        p_id2index[_id] = idx
    
    # workers能力：[0, 1]
    category_size = 7
    sub_category_size = 29
    history_dim = w_num * category_size
    max_entry_num = 661


    p_dim = 2 * max_entry_num
    # workers_quality = (torch.tensor([wws[w_index2id[i]] for i in range(w_num)])) / 100
    workers_history = [[0. for i in range(category_size)] for j in range(w_num)]
    workers_history_count = [[-1 for i in range(category_size)] for j in range(w_num)]
    # quality score withdrawn
    p_history = [torch.zeros(p_dim) for j in range(p_num)]
    p_history_count = [0 for j in range(p_num)]
    
    last_r_state, last_w_state = None, None
    for k, e in enumerate(es):
        print(" %d / %d"%(k, e_num), end='\r')
        w_id = e["worker_id"]
        p_id = e['project_id']
        if not w_id in w_id2index:
            continue
        p = ps[str(p_id)]
        category_feature = one_hot(torch.tensor(p['category']), category_size)
        p_feature = torch.tensor(p_history[p_id2index[str(p_id)]], dtype=torch.float16)
        state, action, reward = torch.cat((category_feature, p_feature), dim=0).float(), torch.tensor([wws[w_id]], dtype=torch.float16), e["withdrawn"] * e['score']

        action = torch.cat((action, torch.tensor(workers_history[w_id2index[w_id]], dtype=torch.float16) / \
                            torch.tensor(workers_history_count[w_id2index[w_id]], dtype=torch.float16)), dim=0)

        # 更新worker history
        workers_history[w_id2index[w_id]][p['category']] += reward
        workers_history_count[w_id2index[w_id]][p['category']] += 1 if workers_history_count[w_id2index[w_id]][p['category']] != -1 else 2

        # 更新p history
        idx = p_history_count[p_id2index[str(p_id)]] * 2

        p_history[p_id2index[str(p_id)]][idx] += e["withdrawn"] * e['score']
        p_history[p_id2index[str(p_id)]][idx+1] += wws[w_id]
        p_history_count[p_id2index[str(p_id)]] += 1
        
        # s1状态就是上一个s2状态
        if last_r_state is None:
            last_r_state = state
            continue

        requester_data['s1'].append(last_r_state)
        requester_data['s2'].append(state)

        last_r_state = state.clone()

        requester_data['a'].append(action.clone())
        requester_data['r'].append(reward)

        worker_data['r'].append(e["withdrawn"] * p['total_awards'] / p['entry_count'])

    requester_data['s1'] = torch.stack(requester_data['s1'], dim=0)
    requester_data['s2'] = torch.stack(requester_data['s2'], dim=0)
    requester_data['a'] = torch.stack(requester_data['a'], dim=0)
    # worker_data['s1'] = torch.stack(worker_data['s1'], dim=0)
    # worker_data['s2'] = torch.stack(worker_data['s2'], dim=0)
    # worker_data['a'] = torch.stack(worker_data['a'], dim=0)

    requester_data['s1'] = (requester_data['s1'] - requester_data['s1'].mean(dim=0)) / (requester_data['s1'].std(dim=0) + 1e-6)
    requester_data['s2'] = (requester_data['s2'] - requester_data['s2'].mean(dim=0)) / (requester_data['s2'].std(dim=0) + 1e-6)
    requester_data['a'] = (requester_data['a'] - requester_data['a'].mean(dim=0)) / (requester_data['a'].std(dim=0) + 1e-6)
    print()

    if print_size:
        for k in ['s1', 's2', 'a']:
            print(("requester %s size: "%k) + str(requester_data[k].size()))
            # print(("worker %s size: "%k) + str(worker_data[k].size()))


    return requester_data, worker_data
       

if __name__ == "__main__":
    dataloader = Dataloader()

    train_data, valid_data, test_data = dataloader.get_datas()


    train_r_data, train_w_data = get_features(train_data, print_size=True)
    valid_r_data, valid_w_data = get_features(valid_data)
    test_r_data, test_w_data = get_features(test_data)

    # de_dim(train_r_data, train_w_data, valid_r_data, valid_w_data, test_r_data, test_w_data)



