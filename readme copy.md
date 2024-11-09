# Long-Term Fairness via Reinforcement Learning (Supplementary Material)

This supplementary information accompanies NeurIPS paper, entitled "Long-Term Fairness via Reinforcement Learning".

## Full Paper (with Appendices)

Our full paper, with all appendices, is included as `full_paper.pdf` in this directory.

## Code

Our code is split into three repositories, each with their own installation / running instructions:

```
.
├── DRL
├── replicator_gym
└── UCB_Fair
```

- DRL contains our implementation of R-TD3. Instructions are provided for assessing the algorithm in our simulated environment.
- replicator_gym implements our simulated environments and greedy baseline agent using the gymnasium API. A script is provided for generating the figures and episodic mean losses for all baseline policies.
- UCB_Fair contains our implementation of the L-UCBFair algorithm. Instructions are provided for assessing the algorithm in our simulated environment.

