无约束优化问题

二阶充分条件：设 f(x) 二阶连续可微，且 ∇f(x∗)=0,∇2f(x∗) 为正定，则 x∗ 是无约束问题的一个严格局部解。 

http://blog.csdn.net/mytestmy/article/details/16903537

最优化方法通常采用迭代的方法求它的最优解，其基本思想是：给定一个初始点

。按照某一迭代规则产生一个点列

，使得当

是有穷点列时，其最后一个点是最优化模型问题的最优解，当是

无穷点列时，它有极限点，且其极限点是最优化模型问题的最优解。一个好的算法应具备的典型特征为：迭代点

能稳定地接近局部极小值点

的邻域，然后迅速收敛于

。当给定的某个收敛准则满足时，迭代即终止。

好吧，上面是一堆描述，来点定义，设![{x_k}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bx_k%7D)为第k次迭代点，![{{\rm{d}}_k}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%7B%5Crm%7Bd%7D%7D_k%7D)第k次搜索方向，![{\alpha _k}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Calpha%20_k%7D)为第k次步长因子，则第k次迭代为

![{x_{k + 1}} = {x_k} + {\alpha _k}{d_k}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bx_%7Bk%20%2B%201%7D%7D%20%3D%20%7Bx_k%7D%20%2B%20%7B%5Calpha%20_k%7D%7Bd_k%7D)                （2）

然后后面的工作几乎都在调整![{\alpha _k}{d_k}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Calpha%20_k%7D%7Bd_k%7D)这个项了，不同的步长![{\alpha _k}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Calpha%20_k%7D)和不同的搜索方向![{{\rm{d}}_k}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%7B%5Crm%7Bd%7D%7D_k%7D)分别构成了不同的方法。



### 最速下降法

最速下降法以负梯度方向作为最优化方法的下降方向，又称为梯度下降法，是最简单实用的方法。Newton法利用二次近似多项式的极值点求法给出原函数极值点的求法