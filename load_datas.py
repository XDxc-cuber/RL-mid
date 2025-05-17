import json
import random

class OurData():
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

        entrys: 按照时间戳排序的list：
            * entry_id: entry的id (int)
            * project_id：所属的project id (int)
            * worker_id：完成这个任务的worker的id (int)
            * score：完成这个任务的worker得到的分数 (int)
            * entry_created_at：entry时间戳 (int)
            * withdrawn：是否拒绝 (int, 0和1)
    """
    def __init__(self, workers: dict, projects: dict, entrys: list):
        self.workers, self.projects, self.entrys = workers, projects, entrys



class Dataloader():

    def __init__(self):
        pass

    def _load_data(self, path):
        with open(path, "r") as f:
            return json.loads(f.readlines()[0])

    def _load_datas(self):
        """
        返回workers, projects, entrys
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

        workers = self._load_data("data/workers.json")
        projects = self._load_data("data/projects.json")
        entrys = self._load_data("data/entrys.json")
        return workers, projects, entrys
    
    def _generate_datas(self, sub_project_ids):
        """
        给定部分projects，选择对应的entrys和所有projects，构造OurData
        """
        entrys = []
        projects = {}
        for p_id in sub_project_ids:
            projects[p_id] = self.projects[p_id]
            for e_id in projects[p_id]["entry_ids"]:
                e_id = str(e_id)
                if not e_id in self.entrys:
                    continue
                entrys.append(self.entrys[e_id])
        entrys.sort(key=lambda x: x["entry_created_at"])

        return OurData(self.workers, projects, entrys)

    def get_datas(self):
        train_rate, valid_rate = 0.8, 0.1

        self.workers, self.projects, self.entrys = self._load_datas()
        train_num = int(len(self.projects) * train_rate)
        valid_num = int(len(self.projects) * valid_rate) + train_num

        p_ids = list(self.projects.keys())

        random.seed(2025)
        random.shuffle(p_ids)
        
        train_data, valid_data, test_data = \
            self._generate_datas(p_ids[:train_num]), \
                self._generate_datas(p_ids[train_num:valid_num]), \
                    self._generate_datas(p_ids[valid_num:])

        return train_data, valid_data, test_data

