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
        features_dim = 0,
        num_layers = 1,
        lr = 0.0001      
    ):
        self.num_head = num_head
        self.embed_dim = embed_dim
        self.features_dim = features_dim
        self.ff_dim = ff_dim
        self.max_len = max_len
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.lr = lr
        self.mlm_model = None
        self.classifier = None
    
    def build(self):
        self.mlm_model = create_masked_language_bert_model(
            self.max_len, self.features_dim, self.vocab_size, self.embed_dim,
            self.num_layers, self.lr, self.num_head, self.ff_dim
        )
    
    def train(self, mlm_ds, epochs = 5):
        self.mlm_model.fit(mlm_ds, epochs=epochs)
    
    def get_path_embeddings(self, encoded_paths, X_positions, X_features):
        self.classifier = self._create_classifier()
        input_data = [encoded_paths, X_positions]
        if X_features != []: input_data.append(X_features)

        return self.classifier.predict(input_data)

    def get_classifier(self):
        return self.classifier

    def _create_classifier(self):
        pretrained_model = tf.keras.Model(
            self.mlm_model.input, 
            self.mlm_model.get_layer("encoder_0/ffn_layernormalization").output
        )

        inputs = layers.Input((self.max_len,), dtype=tf.int64)
        inputs2 = layers.Input((self.max_len,), dtype=tf.int64)
        input_data = [inputs, inputs2]
        if self.features_dim != 0:
            inputs3 = layers.Input((self.features_dim,), dtype=tf.int64)
            input_data.append(inputs3)

        sequence_output = pretrained_model(input_data)
        outputs = layers.GlobalAveragePooling1D()(sequence_output)
        classifier = keras.Model(input_data, outputs, name="classification")

        return classifier

        

