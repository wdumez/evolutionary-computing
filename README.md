# Evolutionary Computing Project

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
- Using k=1 led to random recombination and no real convergence, as expected.
- Using k = 2-10 leads to generally good results. Going higher leads to quicker convergence to a local optimum. Best
  to keep it in this range to really comb the search space. I think k = 5 works best.
- (Lambda+mu) with lambda > mu implicitly does elitism; this is a favorable property to have, especially since we care
  about the best solution we've ever seen! Namely, this best solution is equal to the best solution in each iteration,
  in this case. **IMPORTANT**: this is only true if no mutation is done on the population! If you do this anyway, make
  sure that the best solution (`population[0]`) _does not_ mutate.
- I've been testing, and reading up online. Actually, Python lists are faster than numpy arrays for most things. Numpy
  arrays are only faster for vector operations, like matrix multiplication and dot products and such. Heck, even the
  accessing is O(1). So maybe I should be using lists again? -> Yes!
- Without local search, even tour1000 becomes not-painfully-slow. So I think it's best just to not use it, and instead
  do a more powerful seeding of the initial population.
- Edge crossover is _really_ slow compared to the other operators.
- fitness sharing in both selection and elimination works well; but using both is quite performance impacting. Probably
  best to use only elimination, and offset the lower diversity promotion with a lower selective pressure.
- Inspecting the mutate_func and recombine_func of the candidates after 300 iterations showed that, even when starting
  with inversion and PMX, swap and order became the most common. This implies that swap and order are the best variation
  operators. Also, it shows that this self-adaptivity, though simple, really works well and there is no point not using
  it. It also doesn't really matter with which operators you start, though I would recommend not starting with edge
  crossover because the first iterations will be painfully slow.
- Alpha > 1 does not seem to work very well.
- Be truthful about how you did the hyperparameter search? I mean, it was kind of just trying things that felt good and
  seeing how they worked...
- Self-adaptivity for alpha and sigma does not make sense, because of course the best solutions will want a less diverse
  population in order for them to more likely stay in the population.
- Alpha = 0.5 and sigma = len(distance_matrix) * 0.5 seem to do pretty well, at least for small sizes.
- Now, PMX no longer seems as good as edge crossover...
- Cycle is terrible.
- Edge crossover is far better than the others it seems...
- ! Edge crossover seems to do _nothing_ on tour500 and upwards; switching to order crossover and now it does!
- I feel like sigma creates this kind of inflection point where the mean stagnates; ideally you want sigma to be almost
  0 because then the inflection point is infinitely low, but then you will end up with catastrophic convergence before
  ever reaching it. So I feel like the best thing to do is to have sigma lower over time; e.g. start out at |tour| and
  then lower over time to eventually 1 or some min value.
- So the lectures stated that alpha was more important than sigma, but what I am observing is the opposite.
- I tried letting sigma evolve over time, but this gave the same or worse results than just picking a constant low
  value.
- I tried setting sigma relative to the problem size, but found that a good value of sigma does not seem to scale
  linearly with the problem size. E.g. sigma = 10% of tour50 is not equally as good as 10% of tour500. So I am currently
  just sticking to a constant low value of 5 over all problem sizes and it is working OK.
- Goal: as **exploitative** as possible, while still avoiding catastrophic convergence as much as possible (except for
  tour50 because it is just too easy compared to the higher tours).
- Most important parts of my algorithm: very greedy initialization; diversity promotion (fitness sharing) to avoid cat.
  conv.; elitism to never lose the best solution.
- I tried messing with the greediness, but in the end being greedy all the time gave the best results.
- The greedy initialization depends on the starting city, and it is chosen at random for every member in the population.
  So even if you set the greediness to 1.0, you will still end up with many different solutions. This diversity even
  increases with larger problem sizes, because there are more possible starting cities to choose from (e.g. 50 VS 1000).
- I turned off local search and found that I can do about 2x more iterations in 5 minutes (tour750), yet the rate of
  convergence is virtually identical. So I can get better results this way. This makes sense because local search is
  only really necessary to make your algorithm _more_ exploitative, but I don't need that since my initialization is
  already very exploitative.
- I made local search a lot faster, and it seems to do okay now.
- I sent an email about whether it is alright to do a pure greedy init. Response: it's fine to use a greedy heuristic,
  but also initialize with some randomness. If you can't beat the heuristic, then "I know this feels bad but tough
  luck". So yeah. Best to init. with a greedy seed to at least make it into 200%, then you get 0.8 anyway.
- I think catastrophic convergence with greedy init can be avoided by **not** mutating the population. So right now I am
  doing mutation and local search on the offspring only.
- A good value for sigma_min is not constant in the problem size.
- I need to maintain diversity in the initialization. So far, I assumed that doing a greedy init (1.0) for all
  candidates, that this is not maintained. However, for tour1000 I am seeing that there are *no* duplicates! And the
  avg. (~350) and max. (~550) dist. seems to also imply that it actually *is* kind of diverse...