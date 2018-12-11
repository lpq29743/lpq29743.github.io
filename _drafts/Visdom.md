visdom 是 Pytorch 配套的可视化工具，今天就让我们一起来学习一下。

[visdom 项目地址](https://github.com/facebookresearch/visdom)

visdom 安装和开启：

```bash
# 安装 visdom
pip install visdom
# 开启 visdom，默认端口是 8097，可通过 http://localhost:8097 访问
python -m visdom.server
```

visdom 几个关键概念：

- Panes
- Environments：用户默认有一个名为`main`的`envs`，每个`envs`都可以通过特定的地址（如 http://localhost.com:8097/env/main）访问
- State

简单使用

```python
import torch
import visdom
import numpy as np

# 没有指定 envs 则使用 main
vis = visdom.Visdom(env='test1')
x = torch.arange(1, 30, 0.01)
y = torch.sin(x)
# win 参数指定 pane 的名字，opts 参数添加选项
vis.line(X=x, Y=y, win='sinx', opts={'title': 'y=sin(x)'})
vis.text('Hello World!')
vis.image(np.ones((3, 10, 10))
```

visdom 的 API：

- `vis.scatter` : 2D 或 3D 散点图
- `vis.line` : 线图
- `vis.stem` : 茎叶图
- `vis.heatmap` : 热力图
- `vis.bar` : 条形图
- `vis.histogram`: 直方图
- `vis.boxplot` : 箱型图
- `vis.surf` : 表面图
- `vis.contour` : 轮廓图
- `vis.quiver` : 绘出二维矢量场
- `vis.image` : 图片
- `vis.video`: 视频
- `vis.svg`: SVG 对象
- `vis.text` : 文本
- `vis.mesh` : 网格图
- `vis.save` : 序列化状态

选项（除了图片和文本不可用，其他均可用）

- `options.title` : figure title
- `options.width` : figure width
- `options.height` : figure height
- `options.showlegend` : show legend (`true` or `false`)
- `options.xtype` : type of x-axis (`'linear'` or `'log'`)
- `options.xlabel` : label of x-axis
- `options.xtick` : show ticks on x-axis (`boolean`)
- `options.xtickmin` : first tick on x-axis (`number`)
- `options.xtickmax` : last tick on x-axis (`number`)
- `options.xtickstep` : distances between ticks on x-axis (`number`)
- `options.ytype` : type of y-axis (`'linear'` or `'log'`)
- `options.ylabel` : label of y-axis
- `options.ytick` : show ticks on y-axis (`boolean`)
- `options.ytickmin` : first tick on y-axis (`number`)
- `options.ytickmax` : last tick on y-axis (`number`)
- `options.ytickstep` : distances between ticks on y-axis (`number`)
- `options.marginleft` : left margin (in pixels)
- `options.marginright` : right margin (in pixels)
- `options.margintop` : top margin (in pixels)
- `options.marginbottom`: bottom margin (in pixels)