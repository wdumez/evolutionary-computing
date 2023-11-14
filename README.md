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
  accessing is O(1). So maybe I should be using lists again?

