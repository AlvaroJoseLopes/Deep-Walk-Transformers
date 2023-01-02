from .data import TransductiveDataset, InductiveDataset
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
        self.dataset = None         # TransductiveDataset object
        self.mlm_ds = None          # MLM dataset (Keras Dataset)
        self.mlm_model = None
    
    def fit(
        self,
        G,
        starting_nodes = [],
        batch_size = 128,
        epochs = 5,
        lr = 0.0001,
        standardize = None
    ):
        self.standardize = standardize
        # Build MLM dataset for fake training
        self.dataset = TransductiveDataset(
            self.num_walks, self.walk_len, self.mask_rate, 
            batch_size, standardize
        )
        self.mlm_ds = self.dataset.build(G, starting_nodes)

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
        paths_embeddings = self.mlm_model.get_path_embeddings(
            self.dataset.get_encoded_paths(), self.dataset.get_Xpositions()
        )
        X_paths = self.dataset.get_Xpaths()
        old_mapping = self.dataset.old_mapping

        target_node = 0
        return self._get_embeddings(
            X_paths, old_mapping, paths_embeddings, target_node
        )

    def get_inductive_embeddings(self, G, starting_nodes):
        inductive_ds = InductiveDataset(self.num_walks, self.walk_len, self.standardize)
        encoded_paths, X_positions = inductive_ds.build(G, starting_nodes)
        paths_embeddings = self.mlm_model.get_path_embeddings(
           encoded_paths, X_positions
        )
        X_paths = inductive_ds.get_Xpaths()
        old_mapping = inductive_ds.old_mapping

        target_node = 0
        return self._get_embeddings(
            X_paths, old_mapping, paths_embeddings, target_node
        )
        

    # Some public functions that may be useful
    def get_classifier(self):
        return self.mlm_model.get_classifier()
    

    # def get_Xpaths(self):
    #     return self.dataset.get_Xpaths()

    # # 'Private' functions        
    # def _get_paths_embeddings(self):
    #     return self.mlm_model.get_path_embeddings(
    #         self.dataset.get_encoded_paths(), self.dataset.get_Xpositions()
    #     )

    def _get_embeddings(
        self, X_paths, old_mapping, paths_embeddings, target_node=0
    ):
        node_embeddings = defaultdict(list)
        for walk_index, path in enumerate(X_paths):
            node_embeddings[old_mapping[path[target_node]]].append(paths_embeddings[walk_index])
        
        for target_node in node_embeddings.keys():
            node_embeddings[target_node] = np.array(node_embeddings[target_node]).mean(axis=0)
        
        return node_embeddings
