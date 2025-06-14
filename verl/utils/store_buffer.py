# Copyright 2025 Ziyi Qiu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from verl import DataProto, DataProtoFuture

def check_store_buffer_valid(data: DataProto, info: dict):
    for key, value in info.items():
        assert isinstance(value, np.ndarray), "In store buffer, info should be [str, ndarray]"
        assert len(data) == len(value), "In store buffer, length of info's ndarray should equal length of data"

class StoreBuffer:
    """ Used in sequence-level offpolicy
    
    Args:
        info: could contain elements including ['version': which version generate the rollout]
    """
    def __init__(self, data: DataProto, info: dict):
        check_store_buffer_valid(data, info)
        self.data = data
        self.info = info # should be [str, ndarray], length of ndarray equals data
        
    def __len__(self):
        return len(self.data)
    
    def add(self, new_data: DataProto, new_info: dict):
        check_store_buffer_valid(new_data, new_info)
        if len(self.data) > 0:
            self.data = DataProto.concat([self.data, new_data])
            for key, value in self.info.items():
                assert key in new_info, f'when adding store buffer, key {key} is not in new_info'
                self.info[key] = np.concatenate((value, new_info[key]), axis=0)
        else:
            self.data = new_data
            self.info = new_info
    
    def select(self, batch_size: int, config: dict):
        """ Select one batch for updating
    
        Args:
            config: could contain elements including ['method': selecing method], 
                ['version_ratio': set threshold to select newer rollout, e.g. {`version`: 5, `ratio`: 0.8}]
        """
        if len(self.data) < batch_size:
            return DataProto(), {}
        select_index = []
        replay_index = []
        if "method" not in config or config["method"] == 'naive':
            select_index = [i for i in range(len(self.data) - batch_size, len(self.data))]
            replay_index = [i for i in range(len(self.data) - batch_size)]
        elif config["method"] == 'm_ratio_new':
            assert 'version_ratio' in config, "when using m_ratio_new to select from store buffer, version_ratio is needed in config"
            version_ratio = config["version_ratio"]
            for i in range(len(self.data)):
                if self.info["version"][i] >= version_ratio["version"]:
                    select_index.append(i)
            num_threshold = int(version_ratio["ratio"] * batch_size)
            if len(select_index) >= num_threshold:
                select_index = [select_index[i] for i in range(num_threshold)]
                for i in range(len(self.data)):
                    if not i in select_index:
                        if len(select_index) < batch_size:
                            select_index.append(i)
                        else:
                            replay_index.append(i)
            else:
                return DataProto(), {}
        else:
            raise ValueError(f"`method` element of config should be in [`naive`, `m_ratio_new`]")
        batch, self.data = DataProto.separate_by_index(self.data, select_index, replay_index)
        batch_info = {}
        for key, value in self.info.items():
            batch_info[key] = value[select_index]
            self.info[key] = value[replay_index]
            # print(f"After select, data len={len(self.data)}, info len={len(self.info[key])}")
        return batch, batch_info