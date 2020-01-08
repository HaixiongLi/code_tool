from visdom import Visdom
import numpy as np
import time

# 将窗口类实例化
viz = Visdom(env='Unet')

# 创建窗口并初始化,
viz.line([[0.,0.]], [0], win='train', opts=dict(title='loss&acc', legend=['loss', 'acc']))

# 更新窗口图像
viz.line([[loss, acc]], [global_steps], win='train', update='append')



# 创建窗口并初始化
viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))

# 更新窗口图像
viz.line([loss], [global_steps], win='train_loss', update='append')
