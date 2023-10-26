# Policy and Value Iteration over Frozen Lake Markov Decision Process (MDP) using OpenAI Gym.

A Markov Decision Process (MDP) is a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker. It is defined by a tuple (S, A, P, R, $\gamma$) where:

- $S$ is a finite set of states.
- $A$ is a finite set of actions.
- $P$ is a state transition probability matrix, $P(s'|s, a)$ is the probability that action a in state s at time t will lead to state s' at time t+1.
- $R$ is a reward function, $R(s, a, s')$ is the immediate reward received after transitioning from state s to state s', due to action a.
- $\gamma$ is a discount factor, $\gamma$ ∈ [0, 1].

Here we implement value iteration and policy iteration for the [Frozen Lake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) environment from OpenAI Gym. 

We assume that the stopping tolerance (defined as $\max_s \mid V_{old}(s) - V_{new}(s) \mid $) is tol = $10^{-3}$ and $\gamma = 0.9$.

The state value function in Eqn. below could be used since the state transition probability $p$ is stochastic

$$
V_{k}^{\pi}(s) = \sum_{s',r} p(s',r|s,a)[r + \gamma V_{k-1}^{\pi}(s')], \qquad \forall s \in S
$$

**Policy Iteration**

Policy iteration consists of two steps: policy evaluation and policy improvement. 

1. Policy Evaluation: For a given policy $\pi$, calculate the state-value function $V_{\pi}(s)$ for all states s ∈ S. The state-value function Vπ(s) is calculated as:

$$
V_π(s) = \sum a∈A \pi(a|s) \sum s', r p(s', r|s, a)[r + \gamma V_{\pi}(s')]
$$

2. Policy Improvement: Update the policy based on the current value-function.

$$
\pi'(s) = \argmax_{a} \sum s',r p(s', r|s, a)[r + \gamma V_{\pi}(s')]
$$
These two steps are repeated until the policy converges, i.e., does not change between two consecutive iterations.

**Value Iteration**

Value iteration is a method of computing an optimal MDP policy and its value. 

1. Initialization: Start with an arbitrary value function V(s) and initialize it to zero for all states s ∈ S.

2. Update the value function: For each state s ∈ S, perform the following update:

$$
V(s) = max_a ∑s',r p(s', r|s, a)[r + \gamma V(s')]
$$

3. Check for convergence: Repeat step 2 until the value function converges, i.e., the maximum change in the value function is less than a small positive number ε.

4. Output a deterministic policy, $\pi ≈ \pi*$: 

$$
\pi(s)=\argmax_a \sum s',r p(s', r|s, a)[r + \gamma V(s')]
$$

Value iteration directly finds the optimal value function without having to maintain a policy. Once the optimal value function is found, the optimal policy can be derived from it.