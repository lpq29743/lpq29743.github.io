---
layout: post
title: C++ STL Iterators
categories: C/C++
description: C++ STL Iterators
keywords: C++, STL, Iterators, 迭代器
---

常见的迭代器有以下几种：

1. 输入迭代器（Input Iterator）：只能用来读取指向的值。当迭代器自增时，之前指向的值就不可访问
2. 输出迭代器（Output Iterator）：用于写出的迭代器，其只能增加并将对应内容写入文件中。如果要读取这个迭代中的数据，那么读到的值是未定义的
3. 前向迭代器（Forward Iterator）：类似于输入迭代器，不过其可以在指示范围内迭代多次。就像单向链表一样，只能向前遍历，不能向后遍历，但可以反复迭代
4. 双向迭代器（Bidirectional Iterator）：可以自增，也可以自减，即可以向前或向后迭代
5. 随机访问迭代器（Random Access Iterator）：一次可以跳转到容器的任一元素

##### 输入迭代器

常见的输入迭代器有`std::istream_iterator`，具体使用如下：

```c++
#include <iostream>
#include <iterator>
using namespace std;

int main() {
    // 定义一个输入迭代器接收输入
    istream_iterator<int> a(cin);
    while(1) {
        // 无法执行写操作
        // *a = 3;
        // 执行读操作
        cout << *a << endl;
        // 迭代器自增
        a++;
    }
}
```

##### 输出迭代器

常见的输出迭代器有`std::ostream_iterator`，具体使用如下：

```c++
#include <iostream>
#include <iterator>
using namespace std;

int main() {
    // 定义一个输出迭代器进行输出
    ostream_iterator<int> a(cout, "\n");
    for(int i = 0; i < 5; i++) {
        // 执行写操作
        *a = i;
        // 迭代器自增
        a++;
    }
}
```

##### 前向迭代器

常见的输出迭代器有`std::forward_list`，具体使用如下：

```c++
#include <iostream>
#include <forward_list>
using namespace std;

int main() {
    // 创建一个 forward_list 容器
    forward_list<int> f{1, 2, 3, 4, 5};
    // 使用 forward_list 迭代器
    for(auto i = f.begin(); i != f.end(); i++) {
        cout << *i << endl;
    }
}
```

##### 双向迭代器

常见的输出迭代器有`std::list`、`std::set`和`std::map`，具体使用如下：

```c++
#include <iostream>
#include <list>
using namespace std;

int main() {
    // 创建一个 list 容器
    list<int> l{1, 2, 3};
    // 使用 list 迭代器
    auto i = l.begin();
    // 正向遍历
    while(i != l.end()) {
        cout << *i << endl;
        i++;
    }
    // 反向遍历
    while(i != l.begin()) {
        i--;
        cout << *i << endl;
    }
}
```

##### 随机访问迭代器

常见的输出迭代器有`std::vector`和`std::deque`，具体使用如下：

```c++
#include <iostream>
#include <vector>
using namespace std;

int main() {
    // 创建一个 vector 容器
    vector<int> v{1, 2, 3};
    // 使用 vector 迭代器
    auto i = v.begin();
    // 直接定位
    cout << i[1] << endl;
}
```

