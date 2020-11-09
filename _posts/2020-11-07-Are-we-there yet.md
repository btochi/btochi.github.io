---
layout: post
title: "Are we there yet?"

---

# Introduction

Remember that [last time](http://127.0.0.1:4000/2020/10/22/Variations-on-a-theme-by-Bellman.html) we were left wondering what to do when the state space $$\mathcal{S}$$ was too big. Well, this post is about that.
Once you are done reading this, you'll know how to solve some RL problems with an *infinite* state space. We will again follow the classical [reference](http://www.incompleteideas.net/book/the-book-2nd.html) by Sutton and Barto.

**When faced with complexity one should forego the hope of exact computations and embrace the yoga of estimation.**

We will stop thinking of $$V_\pi,Q_\pi,V_*,Q_*$$ as tables what we "fill in" and instead we will consider them as functions we want to approximate.
As we gain experience our approximations will hopefully improve, and so will our actions. You might think that this is still similar to what we did last time,
but there is a fundamental difference that will become apparent later.

We will propose a parametric approximation

$$\hat{V}_\pi(s,w) \sim V_\pi(s)$$

with $$w$$ in $$\mathbb{R}^d$$. The idea is that $$d$$ should be much smaller that $$\vert\mathcal{S}\vert$$.

# On the search of a good $$w$$

The idea will be simple: after each new interaction with nature we will update $$w$$ in the direction that brings $$\hat{V}_\pi$$ closer to $$V$$ the fastest.
To make this idea precise:

$$w_{t+1} = {w_t} - \frac{1}{2} \alpha \nabla [V_\pi(S_t) - \hat{V}_\pi(S_t,w)]^2  = {w_t} - \frac{1}{2} \alpha [V_\pi(S_t) - \hat{V}_\pi(S_t,w)] \nabla \hat{V}_\pi(s,w)  $$

where $$\alpha$$ is a learning rate that should be fine tuned.

Recall that the gradient points in the direction of fastest growth, so by making $$w_t$$
take a step in the oposite direciton we seek to minimize the distance between $$V_\pi$$ and $$\hat{V}_\pi$$. This idea is usually called *stochastic gradient descent*. Stochastic referst to the fact the we are updating $$w$$ after each interaction with the environment.

**Ok, but we don't actually know $$V_\pi(s)$$! Help!**

[Don't panic](https://en.wikipedia.org/wiki/Phrases_from_The_Hitchhiker%27s_Guide_to_the_Galaxy#Don't_Panic). We will use an approximation for $$V_\pi(s)$$ given by
the information in the current episode, let's call this $$U_t$$.

$$w_{t+1}= {w_t} - \frac{1}{2} \alpha [U_t - \hat{V}_\pi(S_t,w)] \nabla \hat{V}_\pi(S_t,w)  $$

 This is where all the fun RL ideas come along. For example if we wanted to use a Monte Carlo approach, we would set

 $$U_t = G_t = \sum_{k=0}^{\infty} \gamma^{k}R_{t+k}$$

 i.e. the discounted sum of the rewards we observe in the episode. If we wanted to go in a more classical dynammmic programming direction, we would set

$$U_t = E_\pi[R_{t+1}+\gamma \hat{V}(S_t,w)]$$

There are many more ways to define $$U_t$$, each corresponding to a different *classical* RL algorithm.  For the example we will consider later we'll take a look at one named SARSA.

# What should $$\hat{V}$$ be?

Let's see how much mileage we can get out of the simplest possible form for $$\hat{V}$$:

$$\hat{V}(s,w) = \langle x(s),w\rangle$$

where $$x:\mathcal{S}\rightarrow \mathbb{R}^d$$ is a *feature function*. In some way $$x$$ will be the essential tool for solving the dimensionality problem.

Given this choice for $$\hat{V}$$, $$\nabla \hat{V}_\pi(S_t,w) = x(S_t)$$

and our update rule know looks like

$$w_{t+1}= {w_t} - \frac{1}{2} \alpha [U_t - \hat{V}_\pi(S_t,w)] x(S_t)  $$

**I'm losing my patience. How do you define x now?**

There are several options. Let's talk about State Agregation and Tile Coding.

1. State Agregation

Imagine your state space is $$[0,1]^2$$. We could just partition $$[0,1]$$ into $$N$$ subintervals, and then set up a mesh grid by partitioning $$[0,1]^2$$ into $$N^2$$ little squares and
make $$x(s)$$ just indicate on which square $$s$$ lies. $$x(s)$$ would then output a vector consisting of zeros and one 1, indicating the position of $$s$$. This naive idea is enough to make a lot of progress.

2. Tile Coding

The idea here is to refine the method above by translating the mesh around a few times, The little squares each state lies in will then be a better approximation of a neighborhood of the state. Take a look at the picture below, from Sutton Barto.

  ![tile-coding](/assets/tilecoding.jpg)

There's a lot more to say here. In essence we want to reduce the dimension of our state space, while keeping enough information! If you know some math "partitions of unity" might be ringing a bell.

# **SARSA**

Just to make things more fun we will use an update $$U_t$$ that we haven't encountered before: SARSA.

For SARSA we will try to approximate $$Q_*$$ instead of $$\hat{V}_\pi$$. Recall that $$Q_*(s,a)$$ represented the value of taking
action $$a$$ in state $$s$$.

In this case,

$$U_t = R_{t+1}+\gamma \hat{Q}(S_{t+1},A_{t+1},w)$$

We are approximating $$Q_*(s,a)$$ by considering at each time step $$t$$:
1. $$S_t$$ the state we start at.
2. $$A_{t+1}$$ the first action we take
3. $$R_{t+1}$$ the reward we observe after taking the action
4. $$S_{t+1}$$ the new state we observe
5. $$A_{t+1}$$ the second action we take

Get it? SARSA!
Note that we estimate the value of being at $$S_{t+1}$$ and taking $$A_{t+1}$$ by our current *approximation* to $$Q_*$$.

In a similar manner we could define SARSARSA and so on. To ease the tongue, we call these methods *n-step SARSA*. The update would be

$$ U_t = \sum_{k=1}^{n} \gamma^{k-1} R_{t+k} + \gamma^{n}\hat{Q}(S_{t+n},A_{t+n},w_{t+n-1})$$

**Summing up**

We will also start with a linear form $$\hat{Q}(s,a,w) = \langle x(s,a),w\rangle$$, where $$x$$ will be either state aggregation or tile coding (now on the
  space of state-action pairs).

To sum up our n-step SARSA update will look like:

$$w_{t+1}= {w_t} - \frac{1}{2} \alpha \left[\sum_{k=1}^{n} \gamma^{k-1} R_{t+k} + \gamma^{n}\hat{Q}(S_{t+n},A_{t+n},w_{t+n-1}) - \hat{Q}(S_{t},A_{t},w_t)\right]   x(S_{t},A_{t},w_t)  $$


and our algorithm is

  ![sarsa](/assets/n-sarsa.jpg)

(don't worry about $$\epsilon $$ greediness, we'll set $$\epsilon = 0$$, read that line as "we'll take the best action")

# Are we there yet?

Let's get our hands dirty with an example. In the Mountain Car environment our agent is a car that needs to get all the way to the goal on the right.

![set-up](/assets/mountain-car-set-up.jpg)

Each observation is a pair $$(position,velocity)$$ where

$$-1.2 \leq position \leq 0.6$$


$$-0.07 \leq velocity \leq 0.07$$

There are three possible actions: Don't accelerate, accelerate right, accelerate left. This environment is implemented in [OpenAI gym](https://gym.openai.com/).

We receive a reward of $$-1$$ for each time step where we haven't reached the goal. The episode finishes after $$200$$ steps.

**The goal is to reach the right side in less than $$110$$ steps.**

Going forward naively doesn't work!

<video muted autoplay="autoplay" loop="loop" width="768" height="512">
  <source src="/assets/adelante.mp4" type="video/mp4">
</video>

We first trained our agent using state aggregation and one step SARSA.

With a 5 point partition and $$\alpha = 0.1$$ it takes us on average **148.21** steps to reach the goal. Take a look at what this agent is doing:

<video muted autoplay="autoplay" loop="loop" width="768" height="512">
  <source src="/assets/casicasi.mp4" type="video/mp4">
</video>

It looks like we could do better! The agent should learn the right balance between going forward first, then backwards and then forward again.

Finally we trained an agent using 9-step SARSA, a 9 point partition, tile coding with 6 meshes and an appropiate learning rate. It will reach the goal on  **102.5** steps on average, well belove the 110 step threshold!

<video muted autoplay="autoplay" loop="loop" width="768" height="512">
  <source src="/assets/elbueno-car.mp4" type="video/mp4">
</video>


Here's an animation of how our estimate of the cost function (i.e. the amount of steps we estimate we need to reach the goal at each state) changes with each episode. Recall that initially we set it to be zero for every state. Notice how initially we have a "ring", this ring represents the car going back and forth (these are the states that we explore at first!). You can also notice how the cost is lower near the goal and all the way back (since here we could just "let go".)

This is our agent learning!


![animiation](/assets/animation8.gif)



**Remark**

There's a cost to pay for abandoning the idea of computing the exact value of the value function. Before we knew that after changing our policy to be greedy with
respect to some value function, our policy would definitely improve. Since now we are *estimating* the value function, we have lost this assurance, and learning could be chaotic.

You can check out a notebook with the code I used to run this experiments [here!](https://github.com/btochi/ReinforcementLearning/tree/master/MountainCar)
