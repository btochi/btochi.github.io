---
layout: post
title: "Variations on a Theme by Bellman"
---

# Introduction

You are probably familiar with *supervised learning*. The main task here is to estimate a function given labeled data.
For example, you might want to build a model that estimates the market value of a house, recognizes a handwritten digit from an image, or detects spam in your email inbox.
In all of these instances you would first have to collect data and have someone label it for you. Moreover, you or someone else will later
make decisions based on that model.

**What if we tried to incorporate these two key steps in our model?**

This is what Reinforcement Learning (RL) is all about: it's a different incarnation of machine learning.
In RL we try to model how an agent learns by interacting with nature. At each step, our agent will observe the
state of nature, take a decision, and receive a reward.

The meaning of life might be unclear for us humans, but for our
agent it is to maximize the cumulative sum of rewards.

If you are familiar with Economics, this is more or less saying that our agent is a [homo economicus](https://en.wikipedia.org/wiki/Homo_economicus#:~:text=The%20term%20homo%20economicus%2C%20or,their%20subjectively%2Ddefined%20ends%20optimally.&text=In%20game%20theory%2C%20homo%20economicus,the%20assumption%20of%20perfect%20rationality.).

**Why should you care?**

First of all, RL is just very very cool. DeepMind managed to make an agent learn how to [play Atari arcade games](https://www.youtube.com/watch?v=Q70ulPJW3Gk) just from raw pixels. They also reached higher than human game play levels at [Go](https://www.youtube.com/watch?v=WXuK6gekU1Y). For more down to earth applications you might want to google applications of RL to recommender systems, trading, self driving cars and robotics.


**Why do I care?**

I love playing games, and I'm especially fond of Backgammon. In 1992 Gerald Tesauro developed TD-Gammon, an RL based program that played at expert level. It was also
one of the first applications of Neural Networks. Given the complexity of Backgammon (over 10^20 states), this is truly remarkable. My goal is to develop enough theory to understand how and why TD-Gammon worked.

I want to take some time to introduce RL basics, and hopefully get you as hyped as I am. I will be following the book by [Sutton and Barto](http://www.incompleteideas.net/book/the-book-2nd.html).

# Markov Decision process (MDP)

Let's get to precise statements. As shown in the figure, at each time step \\( t\\) the agent observes a state \\( S_t \\), and a reward \\( R_t \\) (the reward
  "corresponds" to the last step) and decides on an action \\( A_t \\). Thus we have a process,

  $$ S_0,A_0,R_1,S_1,A_1,R_2,S_2,A_2,\cdots $$

Let's say that $$\mathcal S,\mathcal R,\mathcal A$$ are the sets of all possible states, rewards and actions.

![agent-environment](/assets/agent-environment.png)

**How is the dynamics determined?**

There is a function, $$p:\mathcal S \times \mathcal A \times \mathcal R \times \mathcal S \rightarrow \mathbb R$$, such that $$p(s',r \vert s,a)$$ is the probability
of landing in state $$s'$$ with reward $$r$$ given that we took action $$a$$ in state $$s$$. You should think of $$p$$ as the laws that rule nature or our fictional environment.

The key aspect here is that we are only conditioning on the last state and action. That is, the past only affects the future through the present. You should think that the state variable has all the information about the past that is relevant. For example, if position were a relevant part of the state variable, we would probabily want speed and acceleration to be part of the state variable too (otherwise we need past states to figure out these other derivatives).  This is called the **Markov Property**.

Consider how flexible this framework is for a moment. For example, Chess can be modelled by letting the final reward be $$1,0.5,0$$ according to whether we win,tie or lose a match. All other rewards should be zero.

In addition, our MDP could finish (like in Chess) or eventually go on forever (like for a self driving car (or life?)).

The goal of our agent is to maximize the sum of the rewards he receives,

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+1+k} $$

Here, $$\gamma$$ is a discount factor. Discounts factors were invented to make infinite sums converge. However, we can also give a nice interpretation: Our agent will appreciate immediate rewards more than delayed rewards the closer $$\gamma$$ is to 0. When $$\gamma$$ is 0, the agent is completely myopic, and won't care
about the consequences of his actions.

Some people live life with $$\gamma = 0$$, are they happier?

# Value Functions

We need some more notation, so grant me some patience. The agent's strategy, plan or *policy* will be formalized as a function $$\pi:\mathcal A \times \mathcal S \rightarrow \mathbb R$$. Here $$\pi(a \vert s)$$ is the probability that the agent takes action $$a$$ in state $$s$$ (yes, our agent's plan need not be deterministic. If you are familiar with game theory this just refers to mixed strategies).

Given a policy $$\pi$$ we can simulate the world and let our agent interact with nature. We would observe random variables $$S_t,R_t,A_t$$ for each time step $$t$$. Since the future is uncertain, and randomness is an intrinsic part of how we modeled the problem, we will be computing expected values to assess the *value* of different situations.

1. $$V_{\pi}(s)=\mathbb E [G_t\vert S_t = s]$$
2. $$Q_{\pi}(s,a)=\mathbb E [G_t\vert S_t = s,A_t=a]$$
3. $$V_{*}(s) =  max_{\pi} V_{\pi}(s)$$
4. $$Q_{*}(s,a) =  max_{\pi} Q_{\pi}(s,a)$$

Equation 1 defines how much reward we should expect to get if we start our simulation at state $$s$$. Equation 2, does a similar thing, if we also assume that
the first action we take is $$a$$. These two functions, $$V_\pi$$ and $$Q_\pi$$ are called the state and state-action value functions for a given policy. Equations 3 and 4 define the optimal value of a state and state-action (i.e. assuming we choose the best policy).

**Why are these functions important?**

Think about how you play chess, backgammon or some board game of your choice. You imagine the future under different moves you could make, and you somehow evaluate
the resulting positions with some heuristic or innate gut feeling. You can think of these equations as the correct way to formalize the idea of evaluating a board position.  If we knew the functions above, we could just take the actions with the highest value, and that would be an optimal policy.

One interesting thing to note, is that by only considering expected values, we are not really assessing risk.

# The Bellman Equation

Now we come to the meaty part of this post. Before we move on, take a minute to think how would yo approach the problem of computing the functions above. Or even more broadly, how would you try to find out an optimal policy.

The Bellman Equations are recursive relations that the value functions satisfy. They impose a condition that actually defines them (i.e. any function satisfying the corresponding Bellman Equation is the corresponding function). Let's deduce the Bellman Equation for $$V_\pi$$

$$\begin{align*}
V_\pi(s) &= \mathbb E [G_t\vert S_t = s]\\
&= \mathbb E [R_{t+1}+\gamma G_{t+1}\vert S_t = s]\\
&=\sum_a \pi(a\vert s) \sum_{s',r} p(s',r\vert s,a) [r+\gamma \mathbb E [G_{t+1}\vert S_{t+1 }= s]\\
&=\sum_a \pi(a\vert s) \sum_{s',r} p(s',r\vert s,a) [r+\gamma\; V_\pi(s')]\\
\end{align*}$$\\

The idea here is to follow the "tree of possibilities" for one time step, and use again the defintion of $$V_\pi$$  . The following "back up diagram" illustrates this idea,

![backup](/assets/back-up-diagram-DP.png)

**Why are these equation important?**

Well, we have one equation and one unknown $$V_\pi(s)$$ for each state $$s$$. This is a linear system, so we could just solve and find $$V_\pi$$. More about this later. To finish this section let me state what the other flavors of the Bellman equation look like.

$$V_{*}(s) = max_a \sum_{s',r} p(s',r\vert s,a) [r+\gamma\; V_{*}(s')]$$

$$Q_{\pi}(s,a) = \sum_{s',r} p(s',r\vert s,a) \sum_{a'} \pi(a'\vert s)[r+\gamma \; Q_\pi(s',a')]$$

$$Q_{*}(s,a) = \sum_{s',r} p(s',r\vert s,a) [r+\gamma\; max_{a'}  Q_{*}(s',a')]$$

You should think what the corresponding back up diagrams look like!

# Dynamic Programming and Policy Iteration

Here is an idea: Initialize a policy $$\pi$$ (randomly or however you want), solve the linear system for $$V_\pi$$ given by the Bellman equations,
and *improve* your policy by defining $$\pi'(a|s) = 1$$ if

$$a = argmax_a \sum_{s',r} p(s',r\vert s,a)[r+\gamma\; V_{\pi}(s')]$$

and $$\pi'(a|s) = 0$$ otherwise. We are *correcting* our policy by choosing in each state the action that leads to the highest expected return (that we estimate so
  far by using $$V_\pi$$).

Repeating this process we will converge to $$V_{*}$$ (this is what we mean by Policy Iteration).

As a side note: think of $$V_{*}$$ as a vector $$x$$. What we are doing is solving
  a linear system that looks like $$x = Ax$$ (i.e. we are finding an eigenvector of eigenvalue $$1$$), by using the [Banach fixed point theorem](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem).

  The whole process looks like this:

$$\pi \rightarrow V_{\pi} \rightarrow \pi' \rightarrow V_{\pi'} \rightarrow \cdots\rightarrow V_{*}$$



**So, have we solved every RL problem?**

Think about what problems we could face in implementing and executing the above algorithm.

1. First of all we might not even know $$p(s',r\vert s,a)$$ Think how complicated this function is for games like Backgammon, or an Atari game.

2. The state space $$\mathcal S$$ might be [impossibly huge](https://en.wikipedia.org/wiki/Game_complexity). Solving a linear system is roughly $$O(n^3)$$, there
are more or less $$10^{20}$$ states in backgammon, so we have no hope of attacking backgammon with this method.

**Ok, what should we do now?**

You should think that almost every RL algorithm, method or idea wants to approximate solving the Bellman equation and do Policy Iteration. The devil of course,
lies in the details.


# Monte Carlo Methods

If $$p$$ represents the laws of the universe or nature, it is a little ambitious to assume we *know* $$p$$. What should we do when we don't know something?

**We learn it.**

One way of learning $$p$$ is to experiment with nature, take notes and estimate from previous experience. This is the whole idea of running a Monte Carlo experiment. Let's take a look at the follwing version of MC from the Sutton and Barto book.

![monte-carlo](/assets/monte-carlo.png)


What's going on above?

1. We initialize a policy, a state-action value function

2. We will run many episodes of our environment, initializing the state and action randomly (initializing the action randomly is important to make sure
  we are exploring our options).

3. We *update* our state-value function for each state we observed (but only for the first occurence of each state in the episode)

4. We *improve* our policy based on the updates state-action value function

I ran this code for computing the optimal strategy for Black Jack, assuming an infinite deck. You can see an animation of how the policy converges to the optimal policy below.


![animation](/assets/animation.gif)


The x-axis shows the dealer's card and the y-axis the sum of the cards in the player's hand. Red means stay and green means hit.
So for example if you get a 10 and a 4, and the dealer shows and ace, you would hit.

You can check out the code [here](https://github.com/btochi/ReinforcementLearning/tree/master/BlackJack)

# Next Steps

**So what about the state space being huge? What should be do then?**

1. First of all, this problem is inherent to RL. It's just how life works, things are complicated.

2. What do mathematicians do when they can't solve a problem? They make some assumptions, and solve a simpler problem. The assumption we will make
is that the value functions are nice enough so that we can approximate them using classical supervised learning methods, like linear regression,
decision trees or neural networks. This is where the fun begins. Tune in next week!
