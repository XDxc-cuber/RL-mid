from load_datas import Dataloader
from feature import get_features





if __name__ == "__main__":
    dataloader = Dataloader()

    train_data, valid_data, test_data = dataloader.get_datas()


    train_r_data, train_w_data = get_features(train_data, print_size=True)
    valid_r_data, valid_w_data = get_features(valid_data)
    test_r_data, test_w_data = get_features(test_data)

    # 最好标准化一下

    
