from hw5.daphne import daphne
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

from bbvi_sampling import BBVISampler

max_time = 3600


def program_1():
    p1_args = {
        "evaluator": "BBVISampling",
        "max_time": max_time,
        "n_bins": 50,
        "bool_res?": False,
        "program_name": f"Program1",
        "use_weights?": False,
        "graph": daphne(['graph', '-i', f'../hw4/programs/1.daphne']),
        "learning_rate": 0.1,
        "batch_size": 100,
        "wandb?": True,
    }

    p1_sampler = BBVISampler(p1_args)
    samples, log_weights, Q = p1_sampler.run()
    samples = np.array(samples[100:])
    weights = np.exp(np.array(log_weights[100:]))

    mean = np.average(samples, weights=weights)
    dist = Q["sample2"]

    print(f"posterior expected value of mu: {mean}")
    print(f"variational distribution of mu: {dist}")
    print(f"elbo max: {np.max(log_weights[100:])}")


def program_2():
    p2_args = {
        "evaluator": "BBVISampling",
        "max_time": max_time,
        "n_bins": 50,
        "bool_res?": False,
        "program_name": f"Program2",
        "use_weights?": False,
        "graph": daphne(['graph', '-i', f'../hw4/programs/2.daphne']),
        "learning_rate": 0.1,
        "batch_size": 100,
        "wandb?": False,
    }

    p2_sampler = BBVISampler(p2_args)
    samples, log_weights, Q = p2_sampler.run()
    samples = np.array([s.numpy() for s in samples])
    slope = samples[100:, 0]
    bias = samples[100:, 1]
    weights = np.exp(np.array(log_weights[100:]))

    slope_mean = np.average(slope, weights=weights)
    bias_mean = np.average(bias, weights=weights)

    print(f"posterior expected value of slope: {slope_mean}")
    print(f"posterior expected value of bias: {bias_mean}")
    print(f"elbo max: {np.max(log_weights[100:])}")


def program_3():
    p3_args = {
        "evaluator": "BBVISampling",
        "max_time": max_time,
        "n_bins": 50,
        "bool_res?": False,
        "program_name": f"Program3",
        "use_weights?": False,
        "graph": daphne(['graph', '-i', f'../hw4/programs/3.daphne']),
        "learning_rate": 0.1,
        "batch_size": 100,
        "wandb?": True,
    }

    p3_sampler = BBVISampler(p3_args)
    samples, log_weights, Q = p3_sampler.run()
    samples = np.array(samples)
    weights = np.exp(np.array(log_weights))
    prob = np.average(np.array([float(s) for s in samples]), weights=np.exp(weights))
    print(f"Posterior probability of z[1] == z[2]: {prob}")


def program_4():
    p4_args = {
        "evaluator": "BBVISampling",
        "max_time": max_time,
        "n_bins": 50,
        "bool_res?": False,
        "program_name": f"Program4",
        "use_weights?": False,
        "graph": daphne(['graph', '-i', f'../hw4/programs/4.daphne']),
        "learning_rate": 0.05,
        "batch_size": 100,
        "wandb?": False,
    }

    p4_sampler = BBVISampler(p4_args)
    samples, log_weights, Q = p4_sampler.run()

    samples = samples[100:]
    log_weights = log_weights[100:]
    weights = np.exp(np.array(log_weights))

    W_0 = np.array([samples[i][0].numpy() for i in range(len(samples))])[:, :, 0].transpose()
    b_0 = np.array([samples[i][1].numpy() for i in range(len(samples))])[:, :, 0].transpose()
    W_1 = np.array([samples[i][2].numpy() for i in range(len(samples))]).reshape(len(samples), 100).transpose()
    b_1 = np.array([samples[i][3].numpy() for i in range(len(samples))])[:, :, 0].transpose()
    results = [W_0, b_0, W_1, b_1]
    labels = ["W_0", "b_0", "W_1", "b_1"]

    for label, result in zip(labels, results):
        dim2 = 1 if label != "W_1" else 10
        exp = np.average(result, axis=1, weights=weights).reshape(10, dim2)
        var = np.var(result, axis=1).reshape(10, dim2)
        plt.figure(figsize=(16, 12))

        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

        sb.heatmap(ax=ax1, data=exp)
        sb.heatmap(ax=ax2, data=var)

        ax1.set_title(f"Expectation of {label}")
        ax2.set_title(f"Variance of {label}")

        plt.savefig(f"plots/Program4_{label}_heatmap.png")
        plt.clf()


def program_5():
    p5_args = {
        "evaluator": "BBVISampling",
        "max_time": max_time,
        "n_bins": 50,
        "bool_res?": False,
        "program_name": f"Program5",
        "use_weights?": False,
        "graph": daphne(['graph', '-i', f'../hw4/programs/5.daphne']),
        "learning_rate": 0.01,
        "batch_size": 100,
        "wandb?": True,
    }

    p5_sampler = BBVISampler(p5_args)
    samples, log_weights, Q = p5_sampler.run()
    print(f"variational distribution of s: {Q}")
    print(f"elbo max: {np.max(log_weights)}")

    pass


if __name__ == "__main__":
    program_1()
    program_2()
    program_3()
    program_4()
    program_5()
