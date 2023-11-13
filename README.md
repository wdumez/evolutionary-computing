# Evolutionary Computing Project

## Targets

- tour50: simple greedy heuristic 27723
- tour100: simple greedy heuristic 90851
- tour200: simple greedy heuristic 39745
- tour500: simple greedy heuristic 157034
- tour750: simple greedy heuristic 197541
- tour1000: simple greedy heuristic 195848

## Observations

- PMX is not as bad as the book makes it out to be.
- Cycle crossover is fully deterministic, so requires higher mutation chance for sufficient exploration.
- Edge crossover is really slow, but improves a lot each iteration.
- Avoid_inf_heuristic gets really slow with large problem sizes.
- Saving the fitness and reusing the results makes for a huge speedup.
- Inversion- and scramble mutation seem to work best.
- Using numpy arrays for memory management is much faster.
- Not doing any mutation leads to catastrophic convergence quite quickly (as expected).
- Always mutating gives good results, though professor said not to do this...
- LSO makes everything really slow, so set to low probability (e.g. 0.01).
- Setting crowding factor really high (100) makes all offspring very similar, not good.
- Lambda plus mu crowding works quite well with visible diversity (mean is higher than best).
- Lambda comma mu crowding doesn't seem to work so well.
- Setting the nr. of mutations too high (10) on inversion mutation gets stuck in a worse local optimum more quickly.
- If lambda > mu, then (lambda+mu)-elim implicitly uses elitism. This is useful when seeding the population with the
  super good heuristic to make sure that effort is not lost.
- When seeding with heuristic solutions, maybe selective pressure should be lowered? Try this.
- Unsurprisingly, the init greedy heuristic is great.
- More stable convergence when only mutating the offspring? Including the population makes it seem less good.
- Crowding _really_ works. You can see the mean bouncing around, whereas without crowding you get catastrophic
  convergence quite quickly.
- Using one of the parents' mutation prob. when recombining doesn't work well, presumably because the best solutions "
  want" to lower the mean mutation prob. so that other solutions don't mutate into something better than them. But this
  makes the prob. near zero, and so there is very little improvement.
- If you think about it, it makes little sense that the mutation prob. is dictated by the most fit candidates. Because
  then it will always become such a way that it only benefits the existing best candidates...
- So having a higher mutation probability (~20% instead of ~5%) makes sense, because at first the recombination is good
  enough to focus down despite the mutation's best efforts, and at the end the higher mutation prob. will cause a chance
  to improve the solution further when a local optimum has been found. Now the only question remains: what's a good
  value for mp?
- Also, maybe it still makes sense for _adaptivity_ of mp, instead of _self-adaptivity_? E.g., mp starts out low but
  increases as time goes on.
- For the recombination and mutation _operators_, I guess it doesn't hurt to allow them to recombine and mutate? All of
  them are somewhat good, and surely the best ones will get used most in the end.
- Using (lambda+mu) with crowding and mu >= lambda causes catastrophic convergence quite quickly (a few hundred
  iterations). This effectively counteracts the diversity promotion (somehow?).

## Things I tried

- I tried adding a version of `mutate_swap` that swaps a nr. of times on average, but this got stuck in a local optimum
  really quickly. Just using the existing function performed much better.

## Results on tour50

### `7995 | mean:        inf | best:   26437.95`

This beat the heuristic! This one was actually _better_ than using edge crossover, even at iteration 5261.

```python
self.array = array
self.fitness = 0.0
self.nr_mutations = 1
self.mutate_func = mutate_inversion
self.recombine_func = recombine_order_crossover
self.local_search_func = local_search_inversion
self.fitness_func = path_length
self.distance_func = distance_hamming

k_selection = 5
k_elimination = 5
crowding_factor = 5
lamda = 100
mu = 40
mutation_prob = 0.05

init_monte_carlo
lamda_plus_mu_crowding
```

### `5261 | mean:        inf | best:   26503.78`

This beat the heuristic!

```python
self.array = array
self.fitness = 0.0
self.nr_mutations = 1
self.mutate_func = mutate_inversion
self.recombine_func = recombine_edge_crossover
self.local_search_func = local_search_inversion
self.fitness_func = path_length
self.distance_func = distance_hamming

k_selection = 5
k_elimination = 5
crowding_factor = 5
lamda = 100
mu = 40
mutation_prob = 0.05

init_monte_carlo
lamda_plus_mu_crowding
```

## Results on tour100

### `2306 | mean:   86574.74 | best:   80950.33`

This beat the heuristic!

```python
self.array = array
self.fitness = 0.0
self.nr_mutations = 1
self.mutate_func = mutate_inversion
self.recombine_func = recombine_edge_crossover
self.local_search_func = local_search_inversion
self.fitness_func = path_length
self.distance_func = distance_hamming

k_selection = 5
k_elimination = 5
crowding_factor = 5
lamda = 100
mu = 40
mutation_prob = 0.05

init_monte_carlo
lamda_plus_mu_crowding
```

## Results on tour200

### `1073 | mean:   66277.00 | best:   58052.85`

Note: this was actually _worse_ at iteration 787 than using edge crossover, but because it's faster it got better
results in the end. Maybe that's worth it?

```python
self.array = array
self.fitness = 0.0
self.nr_mutations = 1
self.mutate_func = mutate_inversion
self.recombine_func = recombine_order_crossover
self.local_search_func = local_search_inversion
self.fitness_func = path_length
self.distance_func = distance_hamming

k_selection = 5
k_elimination = 5
crowding_factor = 5
lamda = 100
mu = 40
mutation_prob = 0.05

init_monte_carlo
lamda_plus_mu_crowding
```

### `787 | mean:   67049.01 | best:   65512.95`

```python
self.array = array
self.fitness = 0.0
self.nr_mutations = 1
self.mutate_func = mutate_inversion
self.recombine_func = recombine_edge_crossover
self.local_search_func = local_search_inversion
self.fitness_func = path_length
self.distance_func = distance_hamming

k_selection = 5
k_elimination = 5
crowding_factor = 5
lamda = 100
mu = 40
mutation_prob = 0.05

init_monte_carlo
lamda_plus_mu_crowding
```

### `6012 | mean:   39244.09 | best:  39244.09`

```python
self.k_selection = 5
self.pop_size = 100
self.offspring_size = 50
self.recombine_probabilities = 0.20
self.recombine_probabilities = 1.0
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
self.recombine_probabilities = 0.20
self.recombine_probabilities = 1.0
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
