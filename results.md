# Benchmark results

## Targets

- tour50: simple greedy heuristic 27723
- tour100: simple greedy heuristic 90851
- tour200: simple greedy heuristic 39745
- tour500: simple greedy heuristic 157034
- tour750: simple greedy heuristic 197541
- tour1000: simple greedy heuristic 195848

Fill-in sheet:

| Best fitness           |   |
|------------------------|---|
| Initialization         |   |
| Selection              |   |
| Mutation operator      |   |
| Recombination operator |   |
| Elimination            |   |
| Local search           |   |
| Adaptivity             |   |
| Self-adaptivity        |   |

## tour50 (27723)

| Final fitness          | 26559.17, 25892.02, 26638.97, 26075.65, 26099.70                            |
|------------------------|-----------------------------------------------------------------------------|
| Initialization         | 1 greedy, 39 non-greedy, 60 random                                          |
| Selection              | K-tournament, k=5                                                           |
| Mutation operator      | Inversion, 20%, only on offspring                                           |
| Recombination operator | Order crossover                                                             |
| Elimination            | Lambda + mu with crowding, lambda = 100, <br/>mu = 40, crowding factor = 10 |
| Local search           | Inversion, length = 2, Hamming distance, after mutation                     |
| Adaptivity             | /                                                                           |
| Self-adaptivity        | Variation operators                                                         |

## tour100 (90851)

## tour200 (39745)

## tour500 (157034)

## tour750 (197541)

## tour1000 (195848)
