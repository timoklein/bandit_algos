# Simple bandits

This repo contains simple implementations of basic bandit algorithms.

## Contents

I use the k-armed testbed from Sutton and Barto's [Reinforcement Learning Book](http://incompleteideas.net/book/the-book.html) for evaluation.
This means the bandits are using Normal distributions with each mean sampled from a Normal distribution and variance set to one.
The number of arms is 10 per default with 1000 trials per run.  
All algorithms from chapter 2 of the book are implemented. Namely:

- **Epsilon Greedy**
- **Upper Confidence Bounds (UCB)**
- **Gradient bandit/Boltzmann Exploration**

Additional algorithms:

- **Thompson Sampling/Probability matching**

## Features

- Pure numpy.
- Typing
- Plotly for visualization of mean rewards and optimal action selection percentages.
