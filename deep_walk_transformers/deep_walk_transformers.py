from data import Dataset
from model import MLMBertModel

class DeepWalkTransformers():
    """
        Deep Walk Transformers model
    """

    def __init__(
        self,
        num_walks,
        walk_len,
        mask_rate
    ):

        self.num_walks = num_walks
        self.walk_len = walk_len
        self.mask_rate = mask_rate
        self.dataset = None
        self.mlm_ds = None          # MLM dataset (Keras Dataset)
        self.mlm_model = None
    
    def fit(
        self,
        G,
        starting_nodes = [],
        batch_size = 128,
        epochs = 5
    ):

        # Build MLM dataset for fake training
        self.dataset = Dataset(self.n_walks, self.walk_len, self.mask_rate)
        self.mlm_ds = self.dataset.build(
            G, starting_nodes, self.mask_rate, standardize=None, batch_size=batch_size
        )

        # Build BERT MODEL
        if self.mlm_model != None: 
            self.mlm_model = MLMBertModel()
            print('Building Masked Language Bert Model ...')
            self.mlm_model.build()
        
        print('Fake Training MLM model ... ')
        self.mlm_model.train(self.mlm_ds, epochs)

    def get_embeddings(self):
        return self.mlm_model.get_node_embeddings(
            self.dataset.get_encoded_paths, self.dataset.get_Xpositions
        )