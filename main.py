from load_datas import Dataloader
from feature import get_features
from dqn_worker import train_worker_dqn
import torch


if __name__ == "__main__":
    dataloader = Dataloader()

    train_data, valid_data, test_data = dataloader.get_datas()

    # train_num, valid_num = 137008, 18079
    # train_r_data, train_w_data, valid_r_data, valid_w_data, test_r_data, test_w_data = {}, {}, {}, {}, {}, {}

    # keys = ['s1', "s2", 'a', 'a_space_emb', 'r']
    # datas = {}
    # for key in keys:
    #     datas[key] = torch.load("data/pca_data/low_dim_%s.pt"%key)

    # for key in keys:
    #     train_r_data[key] = datas[key][:train_num]
    #     valid_r_data[key] = datas[key][train_num:train_num+valid_num]
    #     test_r_data[key] = datas[key][train_num+valid_num:]
        
    # wr = torch.load("data/pca_data/low_dim_w_r.pt")
    # train_w_data['r'] = wr[:train_num]
    # valid_w_data['r'] = wr[train_num:train_num+valid_num]
    # test_w_data['r'] = wr[train_num+valid_num:]


    train_r_data, train_w_data = get_features(train_data)
    valid_r_data, valid_w_data = get_features(valid_data)
    # test_r_data, test_w_data = get_features(test_data)
    # print(train_r_data, train_w_data)
    # 最好标准化一下
    print("开始训练worker的DQN模型...")
    worker_agent = train_worker_dqn(
        train_r_data, 
        valid_data=valid_r_data,
        num_episodes=100,
        batch_size=1000,
        eval_interval=1  # 每个episode都验证
    )
    
    # 保存最终模型
    worker_agent.save_model("models/worker_dqn_final.pth")
    print("模型训练完成并保存")
    
