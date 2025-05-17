from load_datas import Dataloader





if __name__ == "__main__":
    dataloader = Dataloader()

    train_data, valid_data, test_data = dataloader.get_datas()

    print(len(train_data.entrys), len(valid_data.entrys), len(test_data.entrys))
