import glob
import json

import matplotlib.pyplot as plt
import numpy as np

all_params = []
for i in glob.glob("results/*.json"):
    with open(i) as f:
        data = json.load(f)
    all_params.append(data)

with open("toy_params.json") as f:
    ref_params = json.load(f)

x = np.linspace(-5, 5, 1000)
for i in all_params[0]["error"]:
    plt.clf()
    v = np.array([j["value"][i] for j in all_params])
    e = np.array([j["error"][i] for j in all_params])
    pull = (v - ref_params[i]) / e
    mu = np.mean(pull)
    sigma = np.std(pull)
    label = "$\\mu={:.2f}\\pm{:.2f}$\n$\\sigma={:.2f}\\pm{:.2f}$".format(
        mu,
        sigma / np.sqrt(pull.shape[0]),
        sigma,
        sigma / np.sqrt(2 * pull.shape[0]),
    )
    plt.hist(pull, bins=20, range=(-5, 5), label="toy")
    plt.plot(
        x,
        np.exp(-(((x - mu) / sigma) ** 2) / 2)
        / np.sqrt(2 * np.pi)
        / sigma
        * 100
        / 20
        * 10,
        label=label,
    )
    plt.legend()
    plt.title(i)
    plt.savefig(f"results/{i}.png")
