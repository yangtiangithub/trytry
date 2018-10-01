import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 10)
y1 = 2 * x
y2 = x * x

plt.figure(num = "1")
#分图
figure1 = plt.subplot(2, 2, 1)
figure2 = plt.subplot(2, 2, 2)
figure3 = plt.subplot(2, 2, 3)
figure4 = plt.subplot(2,2,4)
#图纸设置
figure1.set_xlim((0, 5))
figure1.set_ylim((0,5))
figure1.set_xlabel("x")
figure1.set_ylabel("y")
figure1.set_title("cool")
#划线 (注意：line名字后面加逗号“,”)
line1, = figure1.plot(x, y1, color = "blue")
line2, = figure2.plot(x, y2, color = "yellow")
scatter1 = figure3.scatter(x, y1)
bar1 = figure4.bar(x, y2)
#图例
figure1.legend(handles = [line1], labels = ("line1",), loc= "best")
figure2.legend(handles = [line2], labels = ("line2",), loc= "best")
figure3.legend(handles = [scatter1], labels = ("scatter1",), loc= "best")
figure4.legend(handles = [bar1], labels = ("bar1",), loc= "best")
plt.show()