Fibonacci 数列在算法中十分常见，其数列形式如下：

> 1, 1, 2, 3, 5, 8, 13, 21, 34, ......

其递归公式可以表示为：
$$
F(n)=F(n-1)+F(n-2) \\
F(0)=1 \\
F(1)=1这篇文章主要围绕它的几种解决展开。
$$
这篇文章主要围绕它的几种解决方法展开。

**递归法**

```python
def fibonacci(n):
    if n == 0 or n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


if __name__ == '__main__':
    print(fibonacci(10))
```

递归法思路简单，但有一定的时间上和空间上的消耗，存在重复计算，容易栈溢出。