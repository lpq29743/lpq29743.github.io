---
layout: post
title: C++ STL Containers
categories: C/C++
description: C++ STL Containers
keywords: C++, STL, Container, 容器
---

C++ STL（Standard Template Library）主要由容器（Containers）、算法（Algorithms）和迭代器（Iterators）三部分组成。这一篇文章主要围绕这容器展开。

C++ 容器主要可分为以下几类：

1. 序列容器（Sequence containers）
2. 容器适配器（Container adapters）
3. 关联容器（Associative containers）

### 序列容器

序列容器按顺序存储元素，常见的序列容器有以下几种

1. vector：无需事先确定大小，支持随机访问；在开头和中间进行增加或删除元素效率很低，时间复杂度为 O(n)
2. list（双向链表）：元素访问速度慢（没有提供 [] 操作符的重载），但增删元素快
3. forward_list（单向列表）：list 的单链表版本
4. deque（双端队列）：可随机存取，在头部和尾部的增删效率高，在中间的效率低，是 vector 和 list 优缺点的结合，使用较少
5. array（数组）：需要实现确定大小，使用较少

#### vector

```c++
#include <iostream>
#include <vector>
using namespace std;

int main() {
    // 创建一个 vector 容器
    vector<int> vec;
    // 往 vector 中添加元素
    vec.push_back(1);
    vec.push_back(2);
    // 获取 vector 长度
    int length = vec.size();
    // 遍历 vector
    for(int i = 0; i < length; i++) {
        // 可用 [] 或 .at() 访问元素
        cout << vec[i] << endl;
    }
    // 从 vector 中移除元素
    vec.pop_back()
}
```

在 vector 中，移除中间元素的最简单做法是删除元素之后，将右边的元素都向左移，这种方法既不美观，又不快捷，因此我们使用下面的方法来使得删除方式更加美观：

```c++
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// 创建一个 vector 容器并将其初始化
vector<int> v{1, 2, 3, 2, 5, 2, 6, 2, 4, 8};

void print() {
	// C++17 下遍历 vector 的方式
    for(auto i: v) {
        cout << i << " ";
    }
    cout << endl;
}

int main() {
    // 抽取要删除的元素
    const auto new_end(remove(begin(v), end(v), 2));
    // 删除元素
    v.erase(new_end, end(v));
    // 打印 vector
    print();
    // 定义谓词函数
    const auto odd([](int i){return i % 2 != 0;});
    // 使用 remove_if 删除元素，谓词函数返回 true 的将不会被删除
    v.erase(remove_if(begin(v), end(v), odd), end(v));
    // 删除后 vector 实例的容量不会变化，将其修改为正确大小
    v.shrink_to_fit();
    // 打印 vector
    print();
}
```

上述的方法虽然使得代码变得美观了，但复杂度仍没有变化，因此我们尝试优化程序，以 O(1) 的时间复杂度删除 vector 中的元素（但会破坏原本 vector 元素的顺序）：

```c++
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

vector<int> v{123, 456, 789, 100, 200};

void print() {
    for(int i: v) {
        cout << i << " ";
    }
    cout << endl;
}

// 实现 O(1) 删除元素，传入 vector 和索引值
template <typename T>
void quick_remove_at(vector<T> &v, size_t idx) {
    // 判断索引值是否合理
    if(idx < v.size()) {
        // 将最后一个元素放在当前位置
        v[idx] = move(v.back());
        // 移除最后一个元素
        v.pop_back();
    }
}

int main() {
    // 按索引移除元素
    quick_remove_at(v, 2);
    output();
    // 按元素值移除元素
    quick_remove_at(v, find(begin(v), end(v), 123));
    print();
}
```

#### list

```c++
#include <iostream>
#include <list>
using namespace std;

int main() {
    // 创建一个 list 容器
    list<int> l;
    // 从前面向 list 中添加元素
    l.push_front(2);
    l.push_front(1);
    // 从后面向 list 中添加元素
    l.push_back(3);
    l.push_back(4);
    // 获取 list 长度
    cout << l.size() << endl;
    // 遍历 list
    for(list<int>::iterator i = l.begin(); i != l.end(); i++) {
        cout << *i << endl;
    }
    // 反向遍历 list
    for(list<int>::reverse_iterator ir = l.rbegin(); ir != l.rend(); ir++) {
        cout << *ir << endl;
    }
}
```

#### deque

```c++
#include <iostream>
#include <deque>
using namespace std;

int main() {
    // 创建一个 deque 容器
    deque<int> d;
    // 从队尾添加元素
    d.push_front(1);
    // 从队头添加元素
    d.push_back(1);
    // 遍历 deque
    for(deque<int>::iterator i = d.begin(); i != d.end(); i++) {
        cout << *i << endl;
    }
}
```

### 容器适配器

容器适配器为序列容器提供了一种不同的接口，常见的容器适配器如下：

- queue（队列）：能够快速删除队头；只能操作队头和队尾
- priority_queue（优先队列）：分为最小值优先队列和最大值优先队列
- stack（栈）：操作简单；但只能操作栈顶元素

#### queue

```c++
#include <iostream>
#include <queue>
using namespace std;

int main() {
    // 创建一个 queue 容器
    queue<int> q;
    // 入队
    q.push(10);
    // 输出队首元素
    cout << q.front() << endl;
    // 输出队尾元素
    cout << q.back() << endl;
    // 出队
    q.pop();
    // 输出队列大小
    cout << q.size() << endl;
}
```

#### priority_queue

```c++
#include <iostream>
#include <priority_queue>
using namespace std;

int main() {
    // 创建一个优先级队列，默认为最大值优先级队列
    priority_queue<int> q1;
    // 创建一个最小值优先级队列
    priority_queue<int, vector<int>, less<int>> q2;
    // 创建一个最大值优先队列
    priority_queue<int, vector<int>, greater<int>> q3;
    // 按最大优先入栈
    q1.push(2);
    q1.push(1);
    q1.push(3);
    // 输出栈顶元素
    cout << q1.top() << endl;
}
```

#### stack

```c++
#include <iostream>
#include <stack>
using namespace std;

int main() {
    // 创建一个 stack 容器
    stack<int> s;
    // 压栈
    s.push(30);
    // 输出栈大小
    cout << s.size() << endl;
    // 输出栈顶元素
    cout << s.top() << endl;
    // 出栈
    s.pop();
}
```

### 关联容器

关联容器的元素位置取决于特定的排序准则，与插入顺序无关，常见的关联容器如下：

- set（集合）：不允许有相同元素
- multiset：允许有相同元素
- map（字典）：不允许有相同键
- multimap：允许有相同键

#### set

```c++
#include <iostream>
#include <set>
using namespace std;

int main() {
    // 创建一个 set 容器
    set<int> s;
    // 插入元素
    s.insert(1);
    s.insert(2);
    // 删除元素
    s.erase(1);
    // 遍历 set
    for(set<int>::iterator i = s.begin(); i != s.end(); i++) {
        cout << *i << endl;
    }
}
```

#### map

```c++
#include <iostream>
#include <map>
#include <string>
using namespace std;

int main() {
    // 创建一个 map 容器
    map<string, int> m;
    // 插入一个键为 Xiaoming，值为 16 的键值对；也可以用来修改值
    m["Xiaoming"] = 16;
    m["Dahong"] = 15;
    // 输出值
    cout << m["Xiaoming"] << endl;
    // 遍历 map
    for(map<string, int>::iterator i = m.begin(); i != m.end(); i++) {
        cout << i->first << " " << i->second << endl;
    }
}
```

在 C++ 中，map 中的键值是不允许修改的，其类型声明使用了`const`。在 C++17 之前，如果要修改键值需要将整个键值对从树中移除，再将其插入，这种操作性能非常低。从 C++17 起，提供了一种性能更高的方式：

```c++
#include <iostream>
#include <map>
using namespace std;

template <typename M>
void print(const M &m) {
    // C++17 下遍历 map 的方式
    for(const auto &[key, value]: m) {
        cout << key << ": " << value << endl;
    }
}

int main() {
    // 创建一个 map 容器并将其初始化
    map<int, string> m {
        {1, "Mike"}, {2, "Louis"}, {3, "Joe"}
    };
    print(m);
    {
        // 利用 C++17 新特性 extract 函数抽取键值对
        auto a(m.extract(1));
        auto a(m.extract(2));
        // 交换键值
        swap(a.key(), b.key());
        // 放回 map
        m.insert(move(a));
        m.insert(move(b));
    }
    print(m);
}
```

