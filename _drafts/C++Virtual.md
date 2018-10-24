定义一个虚函数，是为了允许用基类的指针来调用子类重写的函数，其并不等同于没有具体的实现，没有具体实现的虚函数是纯虚函数，接下来我们通过一个具体的例子来看一看。

如果我们不使用虚函数：

```c++
#include<iostream>
using namespace std;

class Vehicle {
public:
    void GetMaxSpeed() {
        cout << "Unknown" << endl;
    }
};

class Car: public Vehicle {
public:
    void GetMaxSpeed() {
        cout << "12.5" << endl;
    }
private:
    int NumOfSeats;
};

int main() {
    Vehicle *v = new Car();
    v->GetMaxSpeed();
}
```

上述的代码将会打印`Unknown`，但如果我们将函数的定义改为虚函数定义：

```c++
#include<iostream>
using namespace std;

class Vehicle {
public:
    virtual void GetMaxSpeed() {
        cout << "Unknown" << endl;
    }
};

class Car: public Vehicle {
public:
    void GetMaxSpeed() {
        cout << "12.5" << endl;
    }
private:
    int NumOfSeats;
};

int main() {
    Vehicle *v = new Car();
    v->GetMaxSpeed();
}
```

这段程序则会打印出`12.5`。如果我们把虚函数定义为纯虚函数：

```c++
#include<iostream>
using namespace std;

class Vehicle {
public:
    virtual void GetMaxSpeed() = 0;
};

class Car: public Vehicle {
public:
    void GetMaxSpeed() {
        cout << "12.5" << endl;
    }
private:
    int NumOfSeats;
};

int main() {
    Vehicle *v = new Car();
    v->GetMaxSpeed();
}
```

上述函数也能按需求打印。纯虚函数的特点就是要求子类必须重写函数。