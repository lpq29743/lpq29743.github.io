##### 直线

```latex
\begin{tikzpicture}
% draw a line
\draw (0,1) -- (0,0);
% draw two lines
\draw (1,1) -- (1,0) -- (2,0);
% define key properties
\draw[red, dashed, very thick, rotate=30] (4,0) -- (3,0) -- (3,1);
% make the current path to be closed 
\draw (4,0) -- (3,0) -- (3,1) -- cycle;
% connect two points by straight lines that are only horizontal and vertical
\draw (5,0) -| (6,1);
\draw (7,0) |- (8,1);
\end{tikzpicture}
```

##### 箭头

```latex
\begin{tikzpicture}
% arrow with single segment
\draw [->] (0,0) -- (2,0);
\draw [<-] (0, -0.5) -- (2,-0.5);
\draw [|->] (0,-1) -- (2,-1);
% arrow with several segments
\draw [<->] (3,2) -- (3,0) -- (6,0);
\end{tikzpicture}
```

##### 曲线

```latex
\begin{tikzpicture}
// draw a curve
\draw (0,0) .. controls (1,1) .. (4,0)
      (5,0) .. controls (6,0) and (6,1) .. (5,2)
      (6,0) arc [radius=1, start angle=45, end angle= 120];
\end{tikzpicture}
```

