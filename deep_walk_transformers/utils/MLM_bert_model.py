import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

def create_masked_language_bert_model(max_len, vocab_size, embed_dim, num_layers, lr):
    inputs = layers.Input((max_len,), dtype=tf.int64)
    inputs2 = layers.Input((max_len,), dtype=tf.int64)

    word_embeddings = layers.Embedding(
        vocab_size, embed_dim, name="word_embedding"
    )(inputs)
    position_embeddings = layers.Embedding(
        input_dim=max_len,
        output_dim=embed_dim,
        weights=[get_pos_encoding_matrix(max_len, embed_dim)],
        name="position_embedding",
    )(inputs2) # colocando info estrutural aqui
    #)(tf.range(start=0, limit=config.MAX_LEN, delta=1))
    embeddings = word_embeddings + position_embeddings

    encoder_output = embeddings
    for i in range(num_layers):
        encoder_output = bert_module(encoder_output, encoder_output, encoder_output, i)

    mlm_output = layers.Dense(vocab_size, name="mlm_cls", activation="softmax")(
        encoder_output
    )
    mlm_model = MaskedLanguageModel([inputs,inputs2], mlm_output, name="masked_bert_model")

    optimizer = keras.optimizers.Adamax(learning_rate=lr)
    mlm_model.compile(optimizer=optimizer)
    return mlm_model


def bert_module(query, key, value, i, num_head, embed_dim, ff_dim):
    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_head,
        key_dim=embed_dim // num_head,
        name="encoder_{}/multiheadattention".format(i),
    )(query, key, value)
    attention_output = layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
        attention_output
    )
    attention_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ],
        name="encoder_{}/ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc





class MaskedLanguageModel(tf.keras.Model):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    loss_tracker = tf.keras.metrics.Mean(name="loss")   

    def train_step(self, inputs):
        if len(inputs) == 4:
            print("eh pra entrar aqui!!!")
            features, features2, labels, sample_weight = inputs
        else:
            print("aqui soh deve entrar pra fine-tuning!")
            features, features2, labels = inputs
            sample_weight = None

        with tf.GradientTape() as tape:
            predictions = self([features,features2], training=True)
            loss = self.loss_fn(labels, predictions, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker]