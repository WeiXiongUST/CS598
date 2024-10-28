# CS598

# RLHF-Reward-Modeling

## Structure 

The initial release of tis project focus on the Bradley-Terry reward modeling and pairwise preference model. Since then, we have included more advanced techniques to construct preference model. The structure of this project is 

- [`bradley-terry-rm`](./bradley-terry-rm/) to train the classic Bradley-Terry reward model;
- [`pair-pm`](./pair-pm/) to train the pairwise preference model, which takes a prompt and **two responses** as the input and directly predicts the probability of the first response is being preferred;
	- [`SSRM`](./pair-pm/SRRM/): the code of the paper [Semi-Supervised Reward Modeling via Iterative Self-Training](https://arxiv.org/abs/2409.06903)
- [`armo-rm`](./armo-rm/) to train the ArmoRM, which starts with a multi-objective reward model and the reward vector is aggregated by a mixture-of-expert approach in a context-dependent way. See our technical report [[ArmoRM] Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts](https://arxiv.org/abs/2406.12845) for details.


## Installation instructions

It is recommeded to create separate environmnets for the RLHF training and evaluation. Please refer to the corresponding section for detailed instructions.

## Evaluation Results

XX


