import numpy as np
import matplotlib.pyplot as plt


def process_results(program_stream, program_name, evaluator_type, n_bins=50):
    num_samples = 1e4
    samples = []
    for i in range(int(num_samples)):
        samples.append(next(program_stream))
    samples = np.array([s.numpy() for s in samples])

    if samples.ndim == 1:
        print(f"{program_name}_means_from_{evaluator_type}: {np.mean(samples)}")
        plt.hist(samples, bins=n_bins)
        plt.title(f'{program_name}_output_from_{evaluator_type}')
        plt.savefig(f'plots/{program_name}_output_from_{evaluator_type}.png')
        plt.clf()

    elif samples.ndim == 2:
        for j in range(samples.shape[1]):
            print(f"{program_name}_means_for_result_{j}_from_{evaluator_type}: {np.mean(samples[:, j])}")
            plt.hist(samples[:, j], bins=n_bins)
            plt.title(f"{program_name}_output_for_result_{j}_from_{evaluator_type}")
            plt.savefig(f"plots/{program_name}_output_for_result_{j}_from_{evaluator_type}.png")
            plt.clf()

    elif samples.ndim == 3:
        for j in range(samples.shape[1]):
            for k in range(samples.shape[2]):
                print(f"{program_name}_means_for_result_({j},{k})_from_{evaluator_type}: {np.mean(samples[:, j, k])}")
                plt.hist(samples[:, j, k], bins=n_bins)
                plt.title(f"{program_name}_output_for_result_({j},{k})_from_{evaluator_type}")
                plt.savefig(f"plots/{program_name}_output_for_result_({j},{k})_from_{evaluator_type}.png")
                plt.clf()

    else:
        print(f"Plotting for results with {samples.ndim} dimensions is not yet supported.")


def generate_latex_figures():
    pass
