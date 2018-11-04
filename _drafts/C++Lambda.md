```c++
#include <iostream>
using namespace std;

int main() {
    // 用括号形式表示 lambda 表达式
    auto getE([]() {
        return 2.72;
    });
    cout << "E: " << getE() << endl;
    // 用等号形式表示 lambda 表达式
    auto getPI = []() {
        return 3.14;
    };
    cout << "PI: " << getPI() << endl;
    
    // 传入参数
    auto getSum([](auto a, auto b) {
        return a + b;
    });
    cout << "Sum: " << getSum(getE(), getPI()) << endl;
	// 不使用变量保存 lambda 表达式
    cout << "Product: " << [](auto a, auto b) {
        return a * b;
    }
    (getE(), getPI()) << endl;
    
	// 不加 mutable 关键字只可读
    auto Counter1([count = 0]() {
        return count + 1;
    });
    cout << "Counter1: " << Counter1() << endl;
    // 加 mutable 关键字则可读写
    auto Counter2([count = 0]() mutable {return ++count;});
    cout << "Counter2: " << Counter2() << endl;
    
    // 值捕获
    int k1 = 1;
    auto f1([k1]() {
        return k1;
    });
    k1 = 2;
    cout << "Capture by value: " << f1() << endl;
    // 引用捕获
    int k2 = 1;
    auto f2([&k2]() {
        return k2;
    });
    k2 = 2;
    cout << "Capture by reference: " << f2() << endl;
    
	// 捕获外部变量的所有值
    auto getC([ = ](auto r) {
        return 2 * getPI() * r;
    });
    cout << "Circumference: " << getC(5) << endl;
    // // 捕获外部变量的所有引用
    auto getA([&](auto r) {
        return getPI() * r * r;
    });
    cout << "Area: " << getA(5) << endl;
}
```

