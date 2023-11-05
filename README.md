# Evolutionary Computing Project

## Observations

- PMX is not as bad as the book makes it out to be.
- Cycle crossover is fully deterministic, so requires higher mutation chance for sufficient exploration.
- Edge crossover is really slow, but improves a lot each iteration.
- Avoid_inf_heuristic gets really slow with large problem sizes.
- Monte Carlo initialization is awful compared to the heuristic.
- Saving the fitness and reusing the results makes for a huge speedup.
- Inversion- and scramble mutation seem to work best.
- Using numpy arrays for memory management is much faster.
- Using pop size of 10 and off size of 4 gets much better results now?
- Not doing any mutation leads to catastrophic convergence quite quickly (as expected).
- Always mutating gives good results, though professor said not to do this...
- LSO makes everything really slow, so set to low probability (e.g. 0.01).
- Setting k_crowding really high (100) makes all offspring very similar, not good.
- Crowding doesn't seem to be working well: it looks like it has the opposite effect and mean and best converge quickly.
  Maybe I've implemented it wrong?

## Things I tried

- I tried adding a version of `mutate_swap` that swaps a nr. of times on average, but this got stuck in a local optimum
  really quickly. Just using the existing function performed much better.

## Results on tour200

### `6012 | mean:   39244.09 | best:  39244.09`

```python
self.k_selection = 5
self.pop_size = 100
self.offspring_size = 50
self.mutate_chance = 0.20
self.recombine_chance = 1.0
self.lso_chance = 0.0
self.recombine_func = recombine_PMX
self.mutate_func = mutate_inversion
self.lso_func = lso_insert
self.init_func = init_avoid_inf_heuristic
self.select_func = select_k_tournament
self.elim_func = elim_lambda_plus_mu
self.fitness_func = path_length
```

### `4865 | mean:   38662.29 | best:   38662.29`

```python
self.k_selection = 3
self.pop_size = 50
self.offspring_size = 100
self.mutate_chance = 0.20
self.recombine_chance = 1.0
self.lso_chance = 0.0
self.recombine_func = recombine_PMX
self.mutate_func = mutate_inversion
self.lso_func = lso_insert
self.init_func = init_avoid_inf_heuristic
self.select_func = select_k_tournament
self.elim_func = elim_lambda_plus_mu_crowding
self.fitness_func = path_length
self.distance_func = distance
```
