```c++
// 要使用 STL 中的算法函数必须包含头文件 <algorithm>，对于数值算法须包含 <numeric>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;

int main() {
    vector<int> v{1, 2, 3, 1, 1, 5, 2, 4};
    cout << count(v.begin(), v.end(), 2) << endl;
    // 定义谓词函数
    auto greater3([](auto x) {
        return x > 3;
    });
    cout << count_if(v.begin(), v.end(), greater3) << endl;

    auto result = find(v.begin(), v.end(), 6);
    cout << (result == v.end() ? "exist" : "not exist") << endl;
    auto odd([](auto x) {
        return x % 2 == 0;
    });
    // 返回找到的第一个元素的位置
    auto it = find_if(v.begin(), v.end(), odd);
    if(it != v.end()) {
    // 输出元素值
        cout << *it << endl;
    }

    cout << accumulate(v.begin(), v.end(), 0) << endl;
    cout << accumulate(v.begin(), v.end(), 42) << endl;

    // 删除所有相邻的重复元素
    auto end_unique = unique(v.begin(), v.end());
    // unique 不真正删除重复元素，而是把其移到后面，依然保存在原数组中，然后返回去重后最后一个元素的地址，因此需要 erase 删除后面元素
    v.erase(end_unique, v.end());

    vector<int> new_v;
    // 为 copy 操作分配空间
    new_v.resize(v.size());
    copy(v.begin(), v.end(), new_v.begin());
    for(auto i: new_v) {
        cout << i << " ";
    }
    cout << endl;

    sort(v.begin(), v.end());
    for(auto i: v) {
        cout << i << " ";
    }
    cout << endl;
}
```

