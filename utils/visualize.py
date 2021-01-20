import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
import visdom


def display_mask(mask):

    mask = np.where(mask == 1, 255, 0)
    mask = mask.squeeze(axis=0)
    print(type(mask))
    cv2.imshow('mask', mask)



def display_heatmap(heatmap, gt, frame):

    heatmap = heatmap.squeeze()
    gt = gt.squeeze()
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    demo_locatable_axes_easy(ax, heatmap, frame)

    ax_gt = fig.add_subplot(1, 2, 2)
    demo_locatable_axes_easy(ax_gt, gt, frame)


def demo_locatable_axes_easy(ax, join_image, label=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)

    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)

    ax.imshow(label)

    join_image = np.average(np.array(join_image), axis=0)
    join_image = cv2.resize(join_image, (1280, 720))
    im = ax.imshow(join_image, alpha=1)
    # for i in range(image.shape[0]):
    #     im = ax.imshow(image[i, :, :], alpha=0.5)

    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_tick_params(labelright=True)


def display_joints(joints, image):
    colors = [[255, 0, 0], [255, 51, 0], [255, 102, 0], [255, 153, 0], [255, 204, 0], [255, 255, 0], \
              [204, 255, 0], [153, 255, 0], [102, 255, 0], [51, 255, 0], [0, 255, 0], \
              [0, 255, 51], [0, 255, 102], [0, 255, 153], [0, 255, 204], [0, 255, 255], \
              [0, 204, 255], [0, 153, 255], [0, 102, 255], [0, 51, 255], [0, 0, 255], \
              [51, 0, 255], [102, 0, 255], [153, 0, 255], [204, 0, 255]]
    cmap = matplotlib.cm.get_cmap('hsv')
    canvas = image

    for i in range(25):
        rgba = np.array(cmap(1 - i / 18. - 1. / 36))
        rgba[0:3] *= 255
        for j in range(len(joints[i])):
            if joints[i][j][2] > 0.2:
                cv2.circle(
                    canvas,
                    (int(joints[i][j][0] * 15.6), int(joints[i][j][1] * 15.6)),
                    4,
                    colors[i],
                    thickness=-1)

    to_plot = cv2.addWeighted(image, 0.3, canvas, 0.7, 0)
    plt.imshow(to_plot[:, :, [2, 1, 0]])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 12)
    plt.show()


def display_pafs(X, Y, U, V, image):
    plt.figure()
    plt.imshow(image, alpha=0.5)
    s = 10
    if X.ndim == 3:
        for i in range(X.shape[0]):
            Q = plt.quiver(X[i][::s, ::s],
                           Y[i][::s, ::s],
                           U[i][::s, ::s],
                           V[i][::s, ::s],
                           scale=50,
                           headaxislength=4,
                           alpha=.5,
                           width=0.001,
                           color='r')
    else:
        Q = plt.quiver(X[::s, ::s],
                       Y[::s, ::s],
                       U[::s, ::s],
                       V[::s, ::s],
                       scale=50,
                       headaxislength=4,
                       alpha=.5,
                       width=0.001,
                       color='r')


class Visualizer(object):
    def __init__(self, env='defualt', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env='defualt', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]),
                      X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs)
        self.index[name] = x + 1
