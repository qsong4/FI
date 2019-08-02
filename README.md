Dense连接的transformer 是 ps和lm的基础模型
ps 将infer放到了两个multihead attention中间

*35 chicago_x 将_infer 放到了两组mutihead attentnion中间 使用bert position embedding
*35 chicago_x 将_infer 放到了两组mutihead attentnion中间 使用bert position embedding infer中只保留了a_hat❌
*35 chicago_x 将_infer 放到了两组mutihead attentnion中间 使用bert position embedding 每层和前面所有曾链接❌
*35 chicago_x 将_infer 放到了两组mutihead attentnion中间 使用bert position embedding 去掉了infer的ff  acc87.5 5.3
*35 chicago_x 将_infer 放到了两组mutihead attentnion中间 使用bert position embedding 每层和前面所有曾链接 去掉了infer的ff  
