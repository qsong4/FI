Dense连接的transformer 是 ps和lm的基础模型
ps 将infer放到了两个multihead attention中间
lm infer还是放到后面，但是infer中的残差链接变为concat
