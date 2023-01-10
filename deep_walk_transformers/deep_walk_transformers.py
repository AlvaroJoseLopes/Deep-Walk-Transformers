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
        self.features_dim = 0
        self.dataset = None         # TransductiveDataset object
        self.mlm_ds = None          # MLM dataset (Keras Dataset)
        self.mlm_model = None
    
    def fit(
        self,
        G,
        features = None,
        batch_size = 128,
        epochs = 5,
        lr = 0.0001,
        standardize = None
    ):
        if features is not None:
            (_, self.features_dim) = features.shape
        self.standardize = standardize
        # Build MLM dataset for fake training
        self.dataset = TransductiveDataset(
            self.num_walks, self.walk_len, self.mask_rate, 
            batch_size, standardize
        )
        self.dataset.build(G, features)
        self.mlm_ds = self.dataset.get_dataset()

        # Build BERT MODEL
        self.mlm_model = MLMBertModel(
            num_head = self.walk_len, features_dim=self.features_dim, ff_dim = self.embed_dim,
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
            self.dataset.get_encoded_paths(), self.dataset.get_Xpositions(),
            self.dataset.get_Xfeatures()
        )
        X_paths = self.dataset.get_Xpaths()

        target_node = 0
        return self._get_embeddings(
            X_paths, paths_embeddings, target_node
        )

    def get_inductive_embeddings(self, G, starting_nodes, features=None):
        inductive_ds = InductiveDataset(self.num_walks, self.walk_len, self.standardize)
        inductive_ds.build(G, features, starting_nodes)
        encoded_paths, X_positions, X_features = inductive_ds.get_dataset() 
        
        paths_embeddings = self.mlm_model.get_path_embeddings(
           encoded_paths, X_positions, X_features
        )
        X_paths = inductive_ds.get_Xpaths()

        target_node = 0
        return self._get_embeddings(
            X_paths, paths_embeddings, target_node
        )
        

    # Some public functions that may be useful
    def get_classifier(self):
        return self.mlm_model.get_classifier()

    def _get_embeddings(
        self, X_paths, paths_embeddings, target_node=0
    ):
        node_embeddings = defaultdict(list)
        for walk_index, path in enumerate(X_paths):
            node_embeddings[path[target_node]].append(paths_embeddings[walk_index])
        
        for target_node in node_embeddings.keys():
            node_embeddings[target_node] = np.array(node_embeddings[target_node]).mean(axis=0)
        
        return node_embeddings
