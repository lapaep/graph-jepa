import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
import h5py
import numpy as np
import fsspec
import io
from typing import Any
import torch_geometric
import pickle
import warnings
import re
import json
from pathlib import Path
import random


def torch_save(data: Any, path: str) -> None:
    buffer = io.BytesIO()
    torch.save(data, buffer)
    with fsspec.open(path, 'wb') as f:
        f.write(buffer.getvalue())

def torch_load(path: str, map_location: Any = None) -> Any:
    WITH_PT20 = int(torch.__version__.split('.')[0]) >= 2
    WITH_PT24 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 4

    print(f"PyTorch version: {torch.__version__}")
    print(f"WITH_PT24: {WITH_PT24}")

    if WITH_PT24:
        # Code for PyTorch 2.4 and later
        try:
            with fsspec.open(path, 'rb') as f:
                print("Attempting to load with weights_only=True")
                return torch.load(f, map_location, weights_only=True)
        except pickle.UnpicklingError as e:
            error_msg = str(e)
            if "add_safe_globals" in error_msg:
                warnings.warn("Weights only load failed. Retrying without weights_only.")
                with fsspec.open(path, 'rb') as f:
                    return torch.load(f, map_location, weights_only=False)
            else:
                raise e
    else:
        # Fallback for older PyTorch versions (<2.4)
        print("Loading without weights_only (older PyTorch version)")
        with fsspec.open(path, 'rb') as f:
            return torch.load(f, map_location)

class StepTreeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        
        super().__init__(root, transform, pre_transform, pre_filter)
        out = torch_load(self.processed_paths[0])
        
        if not isinstance(out, tuple) or len(out) < 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)
        # if self._data.x is not None:
        #     num_node_attributes = self.num_node_attributes
        #     self._data.x = self._data.x[:, num_node_attributes:]
        # if self._data.edge_attr is not None:
        #     num_edge_attrs = self.num_edge_attributes
        #     self._data.edge_attr = self._data.edge_attr[:, num_edge_attrs:]
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        _raw_file_name = f'{self.root}/raw/steptree_graphs.h5'
        return _raw_file_name

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        sizes=None
        x_lis=[]
        edges_lis=[[],[]]
        y_lis=[]
        slices={
            'x': [0],           # Start-/Endindizes für die Knotenfeatures (node features) der Graphen
            'edge_index': [0],   # Start-/Endindizes für die Kantenliste (edge indices) der Graphen
            'edge_attr': [0],    # Start-/Endindizes für die Kantenattribute (edge features) der Graphen
            'y': [0]  
            }
        
        with h5py.File(self.raw_file_names, 'r') as f:
            length = len(f.keys())

            for key in f.keys():
            
                x_lis.append(f[key]['Nodes'][0:])
                for edge in f[key]['Edges'][0:]:
                    edges_lis[0].append(edge[0]) 
                    edges_lis[1].append(edge[1]) 

                y_lis.append(f[key]['Label'][0].decode("utf-8"))

                slices['x'].append(slices['x'][-1]+len(f[key]['Nodes'][0:]))
                slices['edge_index'].append(slices['edge_index'][-1]+len(f[key]['Edges'][0:]))
                slices['y'].append(slices['y'][-1]+1)
                
        for key in slices:
            slices[key] = torch.tensor(slices[key]).long()

        x = torch.from_numpy(np.concatenate(x_lis, axis=0)).float().view(-1, 1)
    
        edge_index = (torch.from_numpy(np.array(edges_lis))).long()
        class_labels = {string: idx for idx, string in enumerate(set(y_lis))}
        
        with open(f'{self.root}/gg_labels.json', 'w') as json_file:
            json.dump(class_labels, json_file, indent=4)

        y_lis = [class_labels[label] for label in y_lis]

        y = torch.from_numpy(np.array(y_lis)).long()
            
        data = Data(x=x, edge_index=edge_index, y=y)

        self.data, self.slices, sizes = data, slices, sizes

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        assert isinstance(self._data, Data)
        torch_save(
            (self._data.to_dict(), self.slices, sizes, self._data.__class__),
            self.processed_paths[0],
        )

class SMCADDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, cadnet=False):

        self.cadnet = cadnet
        
        super().__init__(root, transform, pre_transform, pre_filter)
        out = torch_load(self.processed_paths[0])
        
        if not isinstance(out, tuple) or len(out) < 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)
        # if self._data.x is not None:
        #     num_node_attributes = self.num_node_attributes
        #     self._data.x = self._data.x[:, num_node_attributes:]
        # if self._data.edge_attr is not None:
        #     num_edge_attrs = self.num_edge_attributes
        #     self._data.edge_attr = self._data.edge_attr[:, num_edge_attrs:]
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        _raw_file_name = f'{self.root}/raw/smcad_graph.h5'
        return _raw_file_name

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        sizes=None
        x_lis=[]
        edge_index_lis=[]
        edge_attr_lis=[]
        y_lis=[]
        slices={
            'x': [0],           # Start-/Endindizes für die Knotenfeatures (node features) der Graphen
            'edge_index': [0],   # Start-/Endindizes für die Kantenliste (edge indices) der Graphen
            'edge_attr': [0],    # Start-/Endindizes für die Kantenattribute (edge features) der Graphen
            'y': [0]  
            }

        with h5py.File(self.raw_file_names, 'r') as f:
            length = len(f.keys())

            for key in f.keys():
                
                if self.cadnet == True and 5 >= len(f[key]['V_1'][0:]):
                    pass
                else:

                    x_lis.append(f[key]['V_1'][0:])
                    
                    edge_index_lis.append(f[key]['A_1_idx'][0:])#+x_len)
                    edge_attr_lis.append(f[key]['V_2'][0:])
                    y_lis.append(f[key]['labels'][0])


                    slices['x'].append(slices['x'][-1]+len(f[key]['V_1'][0:]))
                    slices['edge_index'].append(slices['edge_index'][-1]+len(f[key]['A_1_idx'][0:]))
                    slices['edge_attr'].append(slices['edge_attr'][-1]+len(f[key]['V_2'][0:]))
                    slices['y'].append(slices['y'][-1]+1)
                
        for key in slices:
            slices[key] = torch.tensor(slices[key]).long()

        x = torch.from_numpy(np.vstack(x_lis)).float()

        edge_index = torch.transpose(torch.from_numpy(np.vstack(edge_index_lis)), 0, 1).long()

        edge_attr = torch.from_numpy(np.vstack(edge_attr_lis)).float()

        y = torch.from_numpy(np.array(y_lis)).long()

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        self.data, self.slices, sizes = data, slices, sizes

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        assert isinstance(self._data, Data)
        torch_save(
            (self._data.to_dict(), self.slices, sizes, self._data.__class__),
            self.processed_paths[0],
        )