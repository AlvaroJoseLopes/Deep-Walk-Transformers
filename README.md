# Deep-Walk-Transformers
Node embedding technique based on Masked Language Model.

**A simple example**
----------------------------------------------------------------

You can experiment very easily:

```python
from deep_walk_transformers import DeepWalkTransformers

dwt = DeepWalkTransformers(
    num_walks,
    walk_len,
    mask_rate,
    embed_dim
)
dwt.fit(G, starting_nodes, batch_size, epochs, lr)
node_embeddings = dwt.get_embeddings()
```

See `notebooks` directory for a more complete example.