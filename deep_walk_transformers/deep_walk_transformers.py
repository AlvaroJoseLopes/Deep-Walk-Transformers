from .data import Dataset
from .model import MLMBertModel
from collections import defaultdict
import numpy as np

class DeepWalkTransformers():
    """
        Deep Walk Transformers model
    """

    def __init__(
        self,
        num_walks,
        walk_len,
        mask_rate,
        embed_dim = 32
    ):

        self.num_walks = num_walks
        self.walk_len = walk_len
        self.mask_rate = mask_rate
        self.embed_dim = embed_dim
        self.dataset = None
        self.mlm_ds = None          # MLM dataset (Keras Dataset)
        self.mlm_model = None
    
    def fit(
        self,
        G,
        starting_nodes = [],
        batch_size = 128,
        epochs = 5,
        lr = 0.0001
    ):

        # Build MLM dataset for fake training
        self.dataset = Dataset(self.num_walks, self.walk_len, self.mask_rate)
        self.mlm_ds = self.dataset.build(
            G, starting_nodes, self.mask_rate, standardize=None, batch_size=batch_size
        )

        # Build BERT MODEL
        self.mlm_model = MLMBertModel(
            num_head = self.walk_len, ff_dim = self.embed_dim,
            max_len = self.walk_len, vocab_size = G.number_of_nodes(),
            embed_dim = self.embed_dim, num_layers = 1,
            lr=lr
        )

        print('Building Masked Language Bert Model ...')
        self.mlm_model.build()
        
        print('Fake Training MLM model ... ')
        self.mlm_model.train(self.mlm_ds, epochs)
    
    def get_transductive_embeddings(self):
        paths_embeddings = self._get_paths_embeddings()
        X_paths = self.get_Xpaths()

        target_node = 0
        node_embeddings = defaultdict(list)
        for walk_index, path in enumerate(X_paths):
            node_embeddings[path[target_node]].append(paths_embeddings[walk_index])
        
        for target_node in node_embeddings.keys():
            node_embeddings[target_node] = np.array(node_embeddings[target_node]).mean(axis=0)
        
        return node_embeddings       

    # Some public functions that may be useful
    def get_classifier(self):
        return self.mlm_model.get_classifier()

    def get_Xpaths(self):
        return self.dataset.get_Xpaths()

    # 'Private' functions        
    def _get_paths_embeddings(self):
        return self.mlm_model.get_path_embeddings(
            self.dataset.get_encoded_paths(), self.dataset.get_Xpositions()
        )
