---
layout: post
title: "The DPG Could Be Zero"
date: 2020-01-24
puzzle:
---

"Deep Deterministic Policy Gradient" [(DDPG)](https://arxiv.org/abs/1509.02971) is generally sold as a fairly environment-agnostic algorithm for efficiently learning deterministic policies in MDPs with continuous action spaces. However, the extent to which the true deterministic policy gradient is useful at all for optimization actually depends strongly on the nature of the reward function that you seek to optimize. I'll show below that when the environment dynamics are deterministic (as in [MuJoCo](http://www.mujoco.org/) for instance), **the policy gradient for discrete reward functions vanishes**.

<center><h4>Claim</h4></center>
In a finite-time Markov Decision Process (MDP) with deterministic transition dynamics and a piecewise constant reward function $$r(s, a)$$, the value function $$Q^{\mu_\theta}(s, a)$$ is also piecewise constant for any deterministic policy $$\mu_\theta : \mathcal{S} \rightarrow \mathcal{A}$$, so the [deterministic policy gradient](https://arxiv.org/abs/1509.02971) $$\nabla_\theta J$$ is 0.

$$\nabla_\theta J = \underset{s \sim \rho^\mu}{\mathbb{E}} \left[ \nabla_\theta \mu_\theta(s) \nabla_a Q^{\mu}(s, a) \rvert_{a=\mu_\theta(s)} \right] = 0$$


<center><h4>Proof</h4></center>
Let $$\mathcal{S}_i$$ be the set of states that are $$i$$ steps from terminating under policy $$\mu$$ such that $$\mathcal{S} = \mathcal{S}_1 \cup \mathcal{S}_2 \cup ... \cup \mathcal{S}_n$$. We'll induct on $$i$$.

**Base Case** Consider a terminal state $$s_1 \in \mathcal{S}_1$$. For terminal states, the $$Q$$ function is equal to the reward. Since we assumed the reward function $$r(s,a)$$ is piecewise constant $$\Rightarrow Q(s_1, a)$$ is also piecewise constant.

**Inductive Step** Given that $$Q(s_{i-1}, a)$$ is piecewise constant for all states $$s_{i-1} \in \mathcal{S}_{i-1}$$, we will show that $$Q(s_i, a)$$ is piecewise constant for all $$s_i \in \mathcal{S}_{i}$$. For $$s_i \in \mathcal{S}_i$$, Bellman's equation states

$$ Q^\mu(s_i, a) = \underset{s' \sim T(s_i, a)}{\mathbb{E}} \left[ r(s_i,a) + \gamma \underset{a' \sim \mu(s')}{\mathbb{E}} Q^\mu(s', a') \right].$$

For deterministic transition dynamics and deterministic policies, Bellman's equation reduces to

$$ Q^\mu(s_i, a) = r(s_i,a) + \gamma  Q^\mu(s', \mu(s')) $$

where $$s' = T(s_i, a)$$ so $$s' \in \mathcal{S}_{i-1}$$. Since piecewise constant functions are closed under multiplication and addition, we conclude $$Q^\mu(s_i, a)$$ is piecewise constant.


<!-- #### Proof
"Deep Deterministic Policy Gradient" (DDPG) is generally sold as an algorithm for efficiently learning deterministic policies in MDPs with continuous action spaces. A possibly enlightening observation about the deterministic policy gradient is that it is exactly 0 in a wide range of environments. -->

<!-- "Deep Deterministic Policy Gradient" (DDPG) is an algorithm for efficiently learning deterministic policies in environments with continuous action spaces [(Lillicrap et al, 2016)](https://arxiv.org/abs/1509.02971). When it works, DDPG delivers compelling results. For instance, a cool application of DDPG comes from [Wayve](https://wayve.ai/), an AV startup in London that used DDPG to train a car to drive on curved country roads using only onboard human feedback as a reward signal within minutes [(Kendall et al, 2018)](https://arxiv.org/abs/1807.00412). -->



<!-- In general, however, the community appears to believe that it is difficult to get DDPG to work. In Soft Actor Critic [(Haarnoja et al, 2018)](https://arxiv.org/pdf/1801.01290.pdf), the authors blast DDPG:

<blockquote class="blockquote">
  <center><p class="mb-0">"...a commonly used algorithm in such settings, deep deterministic policy gradient (DDPG), provides for sample-efficient learning but is notoriously challenging to use due to its extreme brittleness and hyperparameter sensitivity."</p>
  <footer class="blockquote-footer">Haarnoja et al, <cite title="Source Title"><a href="https://arxiv.org/pdf/1801.01290.pdf">Soft Actor-Critic</a></cite></footer></center>
</blockquote>

OpenAI offers a similar warning in their spinning up tutorial:

<blockquote class="blockquote">
  <center><p class="mb-0">"While DDPG can achieve great performance sometimes, it is frequently brittle with respect to hyperparameters and other kinds of tuning"</p>
  <footer class="blockquote-footer">OpenAI, <cite title="Source Title"><a href="https://spinningup.openai.com/en/latest/algorithms/td3.html">Spinning Up</a></cite></footer></center>
</blockquote>

The goal of this post is to point out a property of the deterministic policy gradient that is rarely explicitly stated; in the class of environments generally designed for robotics, if the reward function is piecewise constant (meaning the derivative of the reward function is 0 "almost everywhere") the true deterministic policy gradient is 0. 

#### Claim
---
In a finite-time Markov Decision Process (MDP) with continuous action space, deterministic transition dynamics, and a piecewise constant reward function $$r(s, a)$$, the value function $$Q^\pi(s, a)$$ is also piecewise constant for any deterministic policy $$\pi$$, so the deterministic policy gradient is 0 and DDPG reduces to a random walk.

---

#### Toy Example
Before I show the formal proof, here's a simple example which should supply some intuition about why the DPG is sometimes zero and why we might care. Suppose we have a single state MDP with actions $$a \in (-1, 1)$$. The reward is 1 if $$a < 0$$ and 0 if $$a \geq 0$$ (shown below).

{:refdef: style="text-align: center;"}
![reward](/assets/rew.jpg)
{: refdef}

If we apply DDPG to this problem, it does find an optimal deterministic policy. In the top, we show the critic (blue) and the samples from the reward function that it seeks to fit with L2 loss (green dots). In the bottom, we show the actor (black) and the critic (blue) as the actor seeks to find the local maximum with gradient descent. 

{:refdef: style="text-align: center;"}
![reward](/assets/nondiff.gif)
{: refdef}

However, what is the true DPG in this toy problem? For an actor $$\mu$$ parameterized by $$\theta$$, the DPG is given by

$$ \underset{s \sim \rho^\mu}{\mathbb{E}} \nabla_\theta \mu_\theta(s) \nabla_a Q_\phi(s, a) \rvert_{a=\mu_\theta(s)} $$

The value function $$Q^\pi(s, a)$$ is defined as the expected sum of discounted rewards under policy $$\mu$$ starting from state $$s$$ and taking action $$a$$. Since our toy experiment terminates after a single action, the value function is given by

$$ Q^\mu(s, a) = r(s)$$

Therefore, since the reward function is piecewise constant, $$Q$$ is also piecewise constant, and we have

$$ \text{DPG} = \underset{s \sim \rho^\mu}{\mathbb{E}} \nabla_\theta \mu_\theta(s) \underbrace{\nabla_a Q_\phi(s, a) \rvert_{a=\mu_\theta(s)}}_0 = 0 $$

Empirically, DDPG performs well on this toy problem. The reason for its success, however, is not that the critic provided an estimate of the policy gradient to the actor. Instead, the success was somewhat lucky in that the neural network being use to approximate $$Q(s, a)$$ smoothed the reward function in such a way that led the actor to move in a direction of higher reward.


#### Proof
We'll prove the claim is true for all state action pairs by inducting on the number of steps a state is from a terminal state.

**Base Case** Consider a state $$s$$ which is terminal. For all such states, the $$Q$$ function is equal to the reward. Since we assumed the reward function $$r(s,a)$$ is piecewise constant, this implies that $$Q(s, a)$$ is also piecewise constant.

**Inductive Step** Assume that $$Q(s, a)$$ is piecewise constant for all states $$n-1$$ steps from terminating. We will then show that $$Q(s, a)$$ is piecewise constant for states $$n$$ steps from terminating.

Bellman's equation states

$$ Q^\mu(s, a) = \underset{s' \sim T(s, a)}{\mathbb{E}} \left[ r(s,a) + \gamma \underset{a' \sim \mu}{\mathbb{E}} Q^\mu(s', a') \right]$$

For the case of deterministic transition dynamics and deterministic policies, Bellman's equation reduces to

$$ Q^\mu(s, a) = r(s,a) + \gamma  Q^\mu(s', \mu(s')) $$

where $$s' = T(s, a)$$. We assumed $$s$$ was $$n$$ steps from terminating so $$s'$$ must be $$n-1$$ steps from terminating. Since piecewise constant functions are closed under multiplication and addition, we conclude $$Q^\mu(s, a)$$ is piecewise constant.

#### Do We Care?
In general, reinforcement learning algorithms are presented in a way that makes them seem reward function agnostic. Given, the result above, it's somewhat surprising that DDPG works at all on environments such as [MuJoCo](http://www.mujoco.org/) where the transition dynamics are deterministic and the reward functions that are discrete are occasionally used.

Part of the reason DDPG still works in those cases is that the neural surrogate for $$Q(s,a)$$ cannot exactly match a step function. As a result, the critic usually has a small gradient pointed towards regions of higher reward.

It's possibly illuminating to count the number of discrete values that $$Q(s,a)$$ can output. From Bellman's equation, we see that $$Q(s,a)$$ produces $$O(n^t)$$ discrete values where $$n$$ is the number of discrete values produced by $$r(s,a)$$ and state $$s$$ is $$t$$ steps from terminating. -->