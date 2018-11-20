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