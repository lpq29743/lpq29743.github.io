迭代器类型：

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
    isstream_iterator<int> a(cin);
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

