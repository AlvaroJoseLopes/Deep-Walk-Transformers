import walker
from tqdm.notebook import tqdm
import networkx as nx
import numpy as np
from deep_walk_transformers.utils.data_preparation import *

class Dataset():
    """
    Builds Dataset for fake training
    """
    def __init__(
        self,
        n_walks = 50,
        walk_len = 10,
        mask_rate = 0.15
    ):
        self.n_walks = n_walks
        self.walk_len = walk_len
        self.mask_rate = mask_rate
        self.X_paths = []
        self.X_positions = []

    def build(
        self,
        G, 
        start_nodes=None, 
        mask_rate=0.5, 
        standardize=None, 
        batch_size=128
    ):
        self._walk(G, start_nodes)
        x_masked_train, y_masked_labels, sample_weights = self._prepare(
            G.number_of_nodes(), mask_rate, standardize
        )

        mlm_ds = tf.data.Dataset.from_tensor_slices(
            (x_masked_train, self.X_positions, y_masked_labels, sample_weights)
        )

        return mlm_ds.shuffle(1000).batch(batch_size)


    def _walk(self, G, start_nodes=None):
        walks = walker.random_walks(
            G, n_walks=self.n_walks, walk_len=self.walk_len, start_nodes=start_nodes
        )
        print(f'Walks shape: {walks.shape}')

        X_paths = []
        X_positions = []

        print(f'Building X_paths and X_positions ...')
        for walk in tqdm(walks):
            node_target = walk[0] 
            node_positions = []
            for node in walk:
                node_positions.append(
                    nx.shortest_path_length(self.G, source=node_target, target=node)
                )
            
            X_paths.append(walk)
            X_positions.append(node_positions)
        
        self.X_paths = np.array(
            list(map(lambda path: list(map(lambda node: f'node_{str(node)}', path)), X_paths))
        )
        self.X_positions = np.array(X_positions)


    def _prepare(self, vocab_size, mask_rate = 0.5, standardize=None):
        vectorize_layer = get_vectorize_layer(
            self.X_paths,
            vocab_size,
            self.walk_len,
            special_tokens=["[mask]"],
            standardize=standardize
        )
        mask_token_id = vectorize_layer(["[mask]"]).numpy()[0][0]+3
        encoded_paths = encode(self.X_paths, vectorize_layer)
        # x_masked_train, y_masked_labels, sample_weights = 
        return get_masked_input_and_labels(
            encoded_paths, mask_token_id, mask_rate
        )
