import pickle
import matplotlib
import matplotlib.pyplot as plt

hist_list = pickle.load(open('../tmp/mobilenet_v1_hist_list.p', 'rb'))
#plot_list = ['vgg_16/conv1_1','vgg_16/test_double']
fig_i = plt.figure(1)
for i in range(len(hist_list)):
    print(hist_list[i][0])
    plt.scatter(hist_list[i][1][0:-1], hist_list[i][2][0:-1], c='k', marker='o')
    xmin = -0.05*max(hist_list[i][1][0:-1])
    xmax = 1.1*max(hist_list[i][1][0:-1])
    ymax = 1.1*sorted(hist_list[i][2][0:-1])[-2]
    ymin = -(0.1/1.1)*ymax
    plt.xlim(xmin=xmin)
    plt.xlim(xmax=xmax)
    plt.ylim(ymin=ymin)
    plt.ylim(ymax=ymax)
    plt.grid(True)
#      plt.text(0.6*xmax, 0.75*ymax, hist_list[i][0], weight='bold', style='italic', size='larger')
    plt.show()
    plt.close(fig_i)
