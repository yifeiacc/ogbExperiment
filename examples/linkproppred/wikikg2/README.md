# ogbl-wikikg2

**Note (Dec 25, 2020)**: `ogbl-wikikg` is deprecated since negative samples used in validation and test sets are found to be quite biased (i.e., half of the entity nodes are never sampled as negative examples). `ogbl-wikikg2` fixes this issue while retaining everyelse the same. The leaderboard results of `ogbl-wikikg` and `ogbl-wikikg2` are *not* comparable. 

This code includes implementation of TransE, DistMult, ComplEx and RotatE with OGB evaluator. It is based on this [repository](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).

## Training & Evaluation

```
# Run with default config
bash examples.sh
```