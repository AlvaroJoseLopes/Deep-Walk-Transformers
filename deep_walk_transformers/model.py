from deep_walk_transformers.utils.MLM_bert_model import *

class MLMBertModel():
    """
        BERT model for Masked Language Modeling
    """

    def __init__(
        self,
        num_head,
        ff_dim,
        max_len,
        vocab_size,
        embed_dim = 32,
        num_layers = 1,
        lr = 0.0001      
    ):
        self.num_head = num_head
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.max_len = max_len
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.lr = lr
        self.mlm_model = None
        self.classifier = None
    
    def build(self):
        self.mlm_model = create_masked_language_bert_model(
            self.max_len, self.vocab_size, self.embed_dim,
            self.num_layers, self.lr, self.num_head, self.ff_dim
        )
    
    def train(self, mlm_ds, epochs = 5):
        print(f'Training model for {epochs} epochs ...')
        self.mlm_model.fit(mlm_ds, epochs=epochs)
    
    def get_path_embeddings(self, encoded_paths, X_positions):
        self.classifier = self._create_classifier()
        return self.classifier.predict([encoded_paths, X_positions])

    def get_classifier(self):
        return self.classifier

    def _create_classifier(self):
        pretrained_model = tf.keras.Model(
            self.mlm_model.input, 
            self.mlm_model.get_layer("encoder_0/ffn_layernormalization").output
        )

        inputs = layers.Input((self.max_len,), dtype=tf.int64)
        inputs2 = layers.Input((self.max_len,), dtype=tf.int64)
        sequence_output = pretrained_model([inputs,inputs2])
        outputs = layers.GlobalAveragePooling1D()(sequence_output)
        classifier = keras.Model([inputs,inputs2], outputs, name="classification")

        return classifier

        

