强化学习是机器学习的一个分支，近年来受到了广泛关注。

先通过强化学习的几个基本概念了解一下强化学习：

- Agent：通过执行 Action 与 Environment 进行交互，Agent 力图最大化长期累积 Reward
- Action：Agent 可能执行的所有操作的集合，Agent 会在一系列的 Action 中选择一个进行执行
- Discount Factor：越大表示越重视以往经验，越小则越重视眼前利益，一般用于 Q Learning 中
- Environment：Agent 的行动范围。Environment 接收 Agent 的当前状态和执行的 Action 为输入，输出 Agent 的下一状态以及执行 Action 的 Reward
- State：描述特定时间点 Agent 的状况
- Reward：Environment 根据 Agent 执行 Action 的成功和失败情况所作出的反馈。Reward 可以是即时的，也可以是延时的，它有效地评估了 Agent 所执行的 Action
- Policy：描述 Agent 根据当前所处 State 而应采取的 Action 的策略。它将 State 映射成能够取得最大 Reward 的 Action
- Value：与 Reward 表示短期的收益不同，Value 表示的是长期的收益
- Q-value / action-value：与 Value 相似，只不过多带了一个 Action 参数，用于 Q Learning 中
- Trajectory：State 序列以及影响这些 State 的 Action

几个流行的强化学习模型：

##### Q Learning

作为最简单的强化模型，Q Learning 是一个 Value-Based、Off-Policy 的算法。其使用 Q Table 作为查找表，从而求得最大奖励。Q Table 可以表示为 Q(s, a)，每个元素对应每个 State 下执行每个 Action 所得到的 Reward。Q Learning 算法具体的思路如下：

```
Initialize Q(s, a) arbitrarily // 随机初始化 Q Table
Repeat (for each episode): // 以游戏为例，游戏开始到结束就是一个 episode
    Initialize s // 以初始状态开始迭代
    Repeat (for each step of episode): // 迭代一个 episode 的每一个 step
        Choose a from s using policy derived from Q // 使用 Policy（如 ε-greedy）选择 Action
        Take action a, observe r, s' // 观察 Reward 和新 State
        Q(s, a) ← (1 - α) * Q(s, a) + α[r + γmaxQ(s', a')] // 更新 Q Table
        s ← s'	// 更新状态
    until s is terminal // 以游戏为例，游戏结束
```

关于上面描述的算法，有两点需要注意：

1. ε-greedy 策略：每个 State 有 $$\epsilon$$ 的概率随机选取 Action，$$1 - \epsilon$$ 的概率选取当前 State 下 Q 值最大的 Action。$$\epsilon$$ 一般取值较小，0.01 即可
2. Q(s, a) 更新：$$\alpha$$ 参数表示 learning rate，其越大，表示之前训练的效果保留的越少；$$r + γmax_{a'}Q(s', a')$$ 表示估计的总收益，其中 $$r$$ 表示眼前利益，$$max_{a'}Q(s', a')$$ 表示记忆中的利益，指下一状态 $$s'$$ 中能够取得的最大 Reward，$$\gamma$$ 为 Discount Factor，衡量了眼前利益和记忆中的利益的重视偏重。 