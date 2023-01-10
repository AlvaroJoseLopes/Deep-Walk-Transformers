# Deep-Walk-Transformers
Node embedding technique based on Masked Language Model.

**Transductive Embeddings**
----------------------------------------------------------------

You can use the provided additional public method `get_transductive_embeddings()` to obtain the embeddings of the nodes used for training.

```python
from deep_walk_transformers import DeepWalkTransformers

dwt = DeepWalkTransformers(
    num_walks,
    walk_len,
    mask_rate,
    embed_dim
)
dwt.fit(G, starting_nodes, batch_size, epochs, lr)
node_embeddings = dwt.get_transductive_embeddings()
```
**Inductive Embeddings**
----------------------------------------------------------------

Through the `get_inductive_embeddings()` public method you can get the embeddings of nodes unseen on the training data. The required arguments are the full graph `G_full` ( $G_{train}$ + $G_{unseen}$ ) and the unseen nodes `starting_nodes`.

```python
inductive_node_embeddings = dwt.get_inductive_embeddings(G_full, starting_nodes)
```

**Feature Information**
----------------------------------------------------------------
Additionally you can pass the node features as optional argument to the `fit()` and `get_inductive_embeddings()` functions. For the functions:
- **`fit()`**: features argument must be of type `np.ndarray`.
- **`get_inductive_embeddings()`**: features argument must be `dict(np.array)` where the key is the **node id** and the value is the **feature** as `np.array`.

See `notebooks` directory for more complete examples.
