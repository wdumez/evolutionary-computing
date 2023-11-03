# Evolutionary Computing Project

## Observations

- PMX is not as bad as the book makes it out to be.
- Cycle crossover is fully deterministic, so requires higher mutation chance for sufficient exploration.
- Edge crossover is really slow, but improves a lot each iteration.
- Avoid_inf_heuristic gets really slow with large problem sizes.
- Monte Carlo initialization is awful compared to the heuristic.
- Saving the fitness and reusing the results makes for a huge speedup.
- Inversion- and scramble mutation seem to work best.

## Things I tried

- I tried adding a version of `mutate_swap` that swaps a nr. of times on average, but this got stuck in a local optimum
  really quickly. Just using the existing function performed much better.

## Results

### `9276 | mean:   39677.87 | best:  39677.87`

```python
self.k = 5
self.pop_size = 100
self.mu = 20  # Must be even.
self.mutate_chance = 0.20
self.mutation_func = mutate_inversion
self.recombine_func = recombine_PMX
self.fitness_func = path_length
self.init_func = init_avoid_inf_heuristic
self.select_func = select_k_tournament
self.elim_func = elim_lambda_plus_mu
```

### `2162 | mean:   40527.27 | best:  40527.27`

```python
self.k = 5
self.pop_size = 100
self.mu = 50
self.mutate_chance = 0.20
self.mutate_func = mutate_inversion
self.recombine_func = recombine_order_crossover
self.fitness_func = path_length
self.init_func = init_avoid_inf_heuristic
self.select_func = select_k_tournament
self.elim_func = elim_lambda_plus_mu
```