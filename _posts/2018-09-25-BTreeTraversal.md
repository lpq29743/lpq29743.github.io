---
layout: post
title: 二叉树遍历
categories: Algorithm
description: 二叉树遍历
keywords: 算法, 二叉树, 遍历
---

二叉树遍历是面试中经常出现的基础题目，主要需要掌握前序遍历、中序遍历和后序遍历的递归与非递归算法，以及层次遍历算法。

##### 递归实现前序遍历、中序遍历和后序遍历

用递归实现二叉树的这三种遍历方式比较简单，只要按定义撰写具体的代码就可以了。

前序遍历：

```c
void preorder(btree t) {
    if (t) {
        printf("%c ", t->data);
        preorder(t->lchild);
        preorder(r->lchild);
    }
}
```

中序遍历：

```c
void midorder(btree t) {
    if (t) {
        printf("%c ", t->data);
        preorder(t->lchild);
        preorder(r->lchild);
    }
}
```

后序遍历：

```c
void postorder(btree t) {
    if (t) {
        printf("%c ", t->data);
        preorder(t->lchild);
        preorder(r->lchild);
    }
}
```

##### 非递归实现前序遍历、中序遍历和后序遍历

用非递归来实现这三种遍历方式相对于递归方式要复杂一些，需要栈的辅助，而后序遍历则会更复杂一些。

先看前序遍历，实现前序遍历有两种思路。基本思路是：

1. 访问输出结点
2. 将结点入栈
3. 将当前结点设置为原结点的左结点
4. 若没有左结点则进行出栈，设置当前结点为出栈结点的右结点
5. 重复 1 - 4 步，直至栈为空

```c
void preorder(btree t) {
    stack s;
    while(t || s.stop != -1) {
        while(t) {
            printf("%c ", t->data);
            push(&s, t);
            t = t->lchild;
        }
        t = pop(&s);
        t = t->rchild;
    }
}
```

前序遍历还有一种实现方式，但这种方式在中序遍历和后序遍历中并不适用，具体如下：

1. 将根节点入栈
2. 只要栈不为空，出栈并访问，接着依次将访问节点的右节点、左节点入栈

```c
void preorder(btree t) {
    stack s;
    push(&s, t)
    while(s.stop != -1) {
        t = pop(&s);
        printf("%c ", t->data);
        if(t->rchild) {
            push(&t, t->rchild);
        }
        if(t->lchild) {
            push(&t, t->lchild);
        } 
    }
}
```

中序遍历的非递归方式与前序遍历的第一种非递归方式类似，主要的不同在于输出结点的时机不同。具体实现如下：

```c
void midorder(btree t) {
    stack s;
    while(t || s.stop != -1) {
        while(t) {
            printf("%c ", t->data);
            t = t->lchild;
        }
        t = pop(&s);
        push(&s, t);
        t = t->rchild;          
    }
}
```

后序遍历的非递归算法是最难的。相比前序遍历和中序遍历可以先遍历完左子树后再遍历右子树不同，后序遍历需要遍历完左子树和右子树后再访问输出，因此每个结点需要出现在栈顶两次，第一次是为了访问右子树，第二次是为了访问输出。具体如下：

1. 将结点入栈，沿其左子树往下搜索，直到搜索到没有左孩子的结点
2. 此时不能将栈顶结点访问，因此其右子树还要被访问。所以要对其右子树进行 1 的处理
3. 当访问完右子树时，该结点又出现在栈顶，此时可出栈访问

```c
void postorder(btree t) {
    stack s;
    while(t || !s.empty()) {
        while(t) {
            t->isFirst = true;
            push(&s, t);
            t = t->lchild;
        }
        if(!s.empty()) {
            t = s.top();
            s.pop();
            if(t->isFirst == true) {
                t->isFirst = false;
                push(&s, t);
                p = t->rchild;
            } else {
                printf("%c ", t->data);
            }
        }
    }
}
```

##### 层次遍历

如果把前序遍历、中序遍历和后序遍历认为是深度优先遍历的话，那么层次遍历就可以认定为广度优先遍历了。以“深度用栈，广度用队列”的原则，层次遍历需要借助队列的辅助，具体实现如下：

```c
void level(btree t) {
    queue q;
    enter(&q, t);
    while(q.front != q.rear) {
        t = del(&q);
        printf("%c ", t->data);
        if(t->lchild) {
            enter(&q, t->lchild);
        }
        if(t->rchild) {
            enter(&q, t->rchild);
        }
    }
}
```

