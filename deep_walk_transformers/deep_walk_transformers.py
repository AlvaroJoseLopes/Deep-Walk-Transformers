from data import Dataset

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
    
    def fit(
        self,
        G,
        starting_nodes = [],
        batch_size = 128
    ):
        # Build MLM dataset for fake training
        dataset = Dataset(self.n_walks, self.walk_len, self.mask_rate)
        mlm_ds = dataset.build(
            G, self.start_nodes, self.mask_rate, standardize=None, batch_size=batch_size
        )

        # Build BERT MODEL
        # TO-DO


    def get_embeddings():
        pass