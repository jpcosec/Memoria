import matplotlib as mpl
import matplotlib.pyplot as plt


# todo: generalizar input ax,dict
def plot_exp(exps, one_lay=True, f_size=10):
    mpl.style.use('seaborn')
    c = plt.get_cmap('tab20')

    terminal = []
    y_keys = list(exps[2].keys())  # todo:arreglar
    y_keys.remove("epoch")

    f, ax = plt.subplots(1, len(y_keys) + int(one_lay), figsize=(f_size * (len(y_keys) + int(one_lay)), f_size))

    j = 0
    for mod, history in exps.items():
        for i, k in enumerate(y_keys):
            if k != "epoch":
                ax[i].semilogy(history["epoch"], history[k], label=mod, color=c(j / 20))
        terminal.append([int(mod), history["train"][-1], history["test"][-1]])
        j += 1

    ax[1].legend()

    for i, k in enumerate(y_keys):
        if k != "epoch":
            ax[i].set_title(k)
            ax[i].set_xlabel("epochs")
            ax[i].set_ylabel(k)
            ax[i].legend()

    if one_lay:
        ax[-1].loglog([i[0] for i in terminal], [i[1] for i in terminal], label="Train", basex=2)
        ax[-1].loglog([i[0] for i in terminal], [i[2] for i in terminal], label="Test", basex=2)
        ax[-1].semilogx([i[0] for i in terminal], [i[2] / i[1] for i in terminal], label="test/train", basex=2)

        ax[-1].set_title("terminal values")
        ax[-1].set_ylabel("C.E. loss")
        ax[-1].set_xlabel("hidden size")

        ax[-1].legend()
