sort() 函数只适用于 list，其语法如下：

```python
list.sort(cmp=None, key=None, reverse=False)
```

具体参数的意义如下：

- cmp -- 可选参数，传入参数为一个方法，如果指定了该参数会使用该参数的方法进行排序。
- key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。假设元素是元组，可以这样使用：`list.sort(key=lambda x: x[0])`，如果是对象，`list.sort(key=lambda x: x.value)`
- reverse -- 排序规则，reverse = True 降序， reverse = False 升序（默认）。

sorted() 对所有的可迭代序列都有效，与 sort() 有着一样的参数，sorted(list) 不会改变原来的 list，而是会返回一个新的已经排序好的 list。