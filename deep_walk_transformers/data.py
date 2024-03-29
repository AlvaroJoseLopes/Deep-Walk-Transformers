import walker
from tqdm.notebook import tqdm
import networkx as nx
import numpy as np
from deep_walk_transformers.utils.data_preparation import *

class Dataset():
    """
        Base class Dataset for training 
        and getting both transductive and inductive embeddings
    """
    def __init__(
        self,
        n_walks = 50,
        walk_len = 10,
        standardize=None
    ):

        self.n_walks = n_walks
        self.walk_len = walk_len
        self.features = None
        self.X_features = []
        self.walks = None
        self.X_paths = []       # [[1,2], [...], ...]
        self.X_paths_str = []   # ['node_1 node_2', '...', ...]
        self.X_positions = []
        self.encoded_paths = None
        self.standardize = standardize

    def build(self):
        raise NotImplementedError

    def get_dataset(self):
        raise NotImplementedError

    def _walk(self, G, starting_nodes=None):

        self.walks = walker.random_walks(
            G, n_walks=self.n_walks, walk_len=self.walk_len, start_nodes=starting_nodes
        )
        print(f'Walks shape: {self.walks.shape}')

        self.X_paths = []
        self.X_positions = []

        for walk in tqdm(self.walks, desc='Building X_paths and X_positions'):
            node_target = walk[0] 
            node_positions = []
            for node in walk:
                node_positions.append(
                    nx.shortest_path_length(G, source=node_target, target=node)
                )
            
            self.X_paths.append(walk)
            self.X_positions.append(node_positions)
            if self.features is not None:
                self.X_features.append(self.features[node_target])

        
        self.X_paths_str = np.array(
            list(map(lambda path: list(map(lambda node: f'node_{str(node)}', path)), self.X_paths))
        ) # [['node_1', 'node_2']]
        self.X_paths_str = np.array(
            list(map(lambda x: ' '.join(x), self.X_paths_str))
        ) # ['node_1 node_2', ...]

        self.X_positions = np.array(self.X_positions)
        self.X_features = np.array(self.X_features)


    def _prepare(self, vocab_size, special_tokens = None, standardize=None):

        print('Getting Vectorize Layer ...')
        vectorize_layer = get_vectorize_layer(
            self.X_paths_str,
            vocab_size,
            self.walk_len,
            special_tokens=special_tokens,
            standardize=standardize
        )

        self.mask_token_id = vectorize_layer(["[mask]"]).numpy()[0][0]+3

        print('Encoding texts ...')
        self.encoded_paths = encode(self.X_paths_str, vectorize_layer)
    
    def get_encoded_paths(self):
        return self.encoded_paths
    
    def get_Xpositions(self):
        return self.X_positions
    
    def get_Xpaths(self):
        return self.X_paths

    def get_Xfeatures(self):
        return self.X_features

    def get_walks(self):
        return self.walks


class TransductiveDataset(Dataset):
    def __init__(
        self,
        n_walks = 50,
        walk_len = 10,
        mask_rate = 0.15,
        batch_size=128,
        standardize=None
    ):
        super().__init__(n_walks, walk_len, standardize)
        self.mask_rate = mask_rate
        self.mask_token_id = None
        self.batch_size = batch_size
        self.mlm_ds = None
        self.special_tokens = ["[mask]"]

    def build(
        self,
        G, 
        features,
    ):
        self.features = features
        self._walk(G)
        self._prepare(
            G.number_of_nodes(), special_tokens = self.special_tokens,
            standardize = self.standardize
        )

        print(f'Getting masked input (mask token id = {self.mask_token_id}) ...')
        x_masked_train, y_masked_labels, sample_weights = get_masked_input_and_labels(
            self.encoded_paths, self.mask_token_id, self.mask_rate
        )

        dataset = self._get_dataset_tuple(x_masked_train, y_masked_labels, sample_weights)
        
        self.mlm_ds = tf.data.Dataset.from_tensor_slices(dataset)\
                                     .shuffle(1000).batch(self.batch_size)


    def _get_dataset_tuple(self, x_masked_train, y_masked_labels, sample_weights):
        if self.features is not None:
            return (
                x_masked_train, self.X_positions, self.X_features,
                y_masked_labels, sample_weights
            )
        else:
            return (
                x_masked_train, self.X_positions, y_masked_labels, sample_weights
            )
    
    def get_dataset(self):
        return self.mlm_ds


class InductiveDataset(Dataset):
    def __init__(
        self,
        n_walks = 50,
        walk_len = 10,
        standardize = None
    ):
        super().__init__(n_walks, walk_len, standardize)
        self.special_tokens = ["[mask]"]

    def build(
        self,
        G,
        features,
        starting_nodes=None, 
    ):
        self.features = features
        self._walk(G, starting_nodes)
        self._prepare(
            G.number_of_nodes(), special_tokens = self.special_tokens,
            standardize = self.standardize
        )

    def get_dataset(self):
        return self.encoded_paths, self.X_positions, self.X_features 