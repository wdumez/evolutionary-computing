# Benchmark results

## Targets

- tour50: simple greedy heuristic 27723 | 55446
- tour100: simple greedy heuristic 90851 | 181702
- tour200: simple greedy heuristic 39745 | 79490
- tour500: simple greedy heuristic 157034 | 314058
- tour750: simple greedy heuristic 197541 | 395082
- tour1000: simple greedy heuristic 195848 | 391696

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

| Final fitness **[26253.10]**<br/>Beaten by 5.3% | 26559.17, 25892.02, 26638.97, 26075.65, 26099.70                            |
|-------------------------------------------------|-----------------------------------------------------------------------------|
| Initialization                                  | 1 greedy, 39 non-greedy, 60 random                                          |
| Selection                                       | K-tournament, k = 5                                                         |
| Mutation operator                               | Inversion, 20%, only on offspring                                           |
| Recombination operator                          | Order crossover                                                             |
| Elimination                                     | Lambda + mu with crowding, lambda = 100, <br/>mu = 40, crowding factor = 10 |
| Local search                                    | Inversion, length = 2, Hamming distance, after mutation                     |
| Adaptivity                                      | /                                                                           |
| Self-adaptivity                                 | Variation operators                                                         |

## tour100 (90851)

| Final fitness **[84024.18]**<br/>Beaten by 7.5% | 83411.25, 86046.51, 85922.76, 80702.92, 84037.44                            |
|-------------------------------------------------|-----------------------------------------------------------------------------|
| Initialization                                  | 1 greedy, 39 non-greedy, 60 random                                          |
| Selection                                       | K-tournament, k = 5                                                         |
| Mutation operator                               | Inversion, 20%, only on offspring                                           |
| Recombination operator                          | Order crossover                                                             |
| Elimination                                     | Lambda + mu with crowding, lambda = 100, <br/>mu = 40, crowding factor = 10 |
| Local search                                    | Inversion, length = 2, Hamming distance, after mutation                     |
| Adaptivity                                      | /                                                                           |
| Self-adaptivity                                 | Variation operators                                                         |

## tour200 (39745)

41070.61199512355, 41348.24915661073, 40783.63353951755, 39381.8947916869, 41166.9970669643

| Final fitness **[40750.28]**<br/>Lost by 2.5% | 41070.61, 41348.25, 40783.63, 39381.89, 41167.00                            |
|-----------------------------------------------|-----------------------------------------------------------------------------|
| Initialization                                | 1 greedy, 39 non-greedy, 60 random                                          |
| Selection                                     | K-tournament, k = 5                                                         |
| Mutation operator                             | Inversion, 20%, only on offspring                                           |
| Recombination operator                        | Order crossover                                                             |
| Elimination                                   | Lambda + mu with crowding, lambda = 100, <br/>mu = 40, crowding factor = 10 |
| Local search                                  | Inversion, length = 2, Hamming distance, after mutation                     |
| Adaptivity                                    | /                                                                           |
| Self-adaptivity                               | Variation operators                                                         |

## tour500 (157034)

## tour750 (197541)

## tour1000 (195848)
