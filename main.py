from load_datas import Dataloader
from feature import get_features
from dqn_worker import train_worker_dqn

if __name__ == "__main__":
    dataloader = Dataloader()

    train_data, valid_data, test_data = dataloader.get_datas()


    train_r_data, train_w_data = get_features(train_data, print_size=True)
    valid_r_data, valid_w_data = get_features(valid_data)
    test_r_data, test_w_data = get_features(test_data)

<<<<<<< HEAD
    # 最好标准化一下
=======
    # 训练worker的DQN模型
    print("开始训练worker的DQN模型...")
    worker_agent = train_worker_dqn(
        train_w_data, 
        valid_data=valid_w_data,
        num_episodes=2,
        eval_interval=1  # 每个episode都验证
    )
    
    # 保存最终模型
    worker_agent.save_model("models/worker_dqn_final.pth")
    print("模型训练完成并保存")
>>>>>>> 05be04cb61c64d747b7e22ff92fea9278ed41a6c

    
