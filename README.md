# Evolutionary Computing Project

## Observations

- PMX is not as bad as the book makes it out to be.
- Cycle crossover is fully deterministic, so requires higher mutation chance for sufficient exploration.
- Edge crossover is really slow, but improves a lot each iteration.
- Avoid_inf_heuristic gets really slow with large problem sizes.
- Monte Carlo initialization is awful compared to the heuristic.
- Saving the fitness and reusing the results makes for a huge speedup.

## Things I tried

- I tried adding a version of `mutate_swap` that swaps a nr. of times on average, but this got stuck in a local optimum
  really quickly. Just using the existing function performed much better.

## Results

### `9276 | mean:   39677.87 | best:  39677.87`

with:

```python
self.k_in_selection = 5
self.population = []
self.population_size = 100
self.nr_offspring = 20  # Must be even.
self.mutate_chance = 0.20
self.mutation_function = mutate_inversion
self.recombine_function = recombine_PMX
self.fitness_function = fitness_length
self.init_function = init_avoid_inf_heuristic
self.select_function = select_k_tournament
```

Testing