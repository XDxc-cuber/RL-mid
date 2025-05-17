import torch
import torch.nn as nn
from load_datas import OurData, Dataloader

"""
注意：所有dict的key都是str

workers: {key: worker_id, value: "worker_quality"}

projects: key为project_id，value为：
    * sub_category：项目子类别 (0到N的int)
    * category：项目大类 (0到N的int)
    * entry_count：项目entry的数目，相当于子任务 (int)
    * entry_ids：每个entry的id (int)
    * client_feedback：请求者的反馈分数 (float)
    * average_score：所有entry的平均分 (float)
    * total_awards：项目给出的报酬 (float)
    * start_date：开始时间 (int)
    * deadline：结束时间 (int)

entrys: 按照时间戳排序的dict，key为e_id，value：
    * entry_id: entry的id (int)
    * project_id：所属的project id (int)
    * worker_id：完成这个任务的worker的id (int)
    * score：完成这个任务的worker得到的分数 (int)
    * entry_created_at：entry时间戳 (int)
    * withdrawn：是否拒绝 (int, 0和1)
    """
def get_features(data: OurData):
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

    w_num = len(ws)
    e_num = len(es)
    w_id2index, w_index2id = {}, list(ws.keys())
    for idx, _id in enumerate(w_index2id):
        w_id2index[_id] = idx
    
    # workers能力：[0, 1]
    category_size = 7
    sub_category_size = 29
    history_dim = w_num * category_size
    workers_quality = (torch.tensor([ws[w_index2id[i]] for i in range(w_num)]) + 1 ) / 101
    workers_history = [0. for i in range(history_dim)]
    workers_history_count = [-1 for i in range(history_dim)]
    
    # last_state = None
    for k, e in enumerate(es):
        print("%d / %d"%(k, e_num), end='\r')
        w_id = str(e["worker_id"])
        if not w_id in w_id2index:
            continue
        state, action, reward = [], w_id2index[w_id], e["withdrawn"] * e['score']
        p = ps[str(e['project_id'])]

         # 更新worker history
        updated_index = int(action * category_size + p['category'])
        workers_history[updated_index] += reward
        workers_history_count[updated_index] += 1 if workers_history_count[updated_index] != -1 else 2
        
        # 得到s2状态
        state.append(workers_quality.clone())
        state.append(torch.tensor(workers_history) / torch.tensor(workers_history_count))
        state.append(torch.tensor([p['category'], p['sub_category']]))
        state = torch.cat(state, 0).type_as(torch.zeros(1, dtype=torch.float16))
        
        # s1状态就是上一个s2状态
        # if last_state is None:
        #     last_state = state
        #     continue
        # requester_data['s1'].append(last_state)
        # worker_data['s1'].append(last_state.clone())
        requester_data['s2'].append(state)
        worker_data['s2'].append(state.clone())
        last_state = state.clone()
        requester_data['a'].append(action)
        requester_data['r'].append(reward)

        worker_data['a'].append(e["withdrawn"])
        worker_data['r'].append(e["withdrawn"] * p['total_awards'] / p['entry_count'])

    # requester_data['s1'] = torch.stack(requester_data['s1'], dim=0)
    requester_data['s2'] = torch.stack(requester_data['s2'], dim=0)
    # worker_data['s1'] = torch.stack(worker_data['s1'], dim=0)
    worker_data['s2'] = torch.stack(worker_data['s2'], dim=0)
    print()

    return requester_data, worker_data
       

if __name__ == "__main__":
    dataloader = Dataloader()

    train_data, valid_data, test_data = dataloader.get_datas()


    train_r_data, train_w_data = get_features(train_data)
    valid_r_data, valid_w_data = get_features(valid_data)
    test_r_data, test_w_data = get_features(test_data)






