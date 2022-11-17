import numpy as np
from scipy import stats
import torch
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

"""Load data"""

train_dataset = CIFAR10(root='data/', download=True, train=True)
X = torch.flatten(torch.tensor(train_dataset.data), start_dim=1, end_dim=-2) / 255
X = np.array(torch.mean(X, axis=1))
X.shape

classes = train_dataset.classes
Y = []
class_count = {}
for _, index in train_dataset:
    Y.append(index)
    
Y = np.array(Y)
Y.shape

n = X.shape[0] # total number of training images
K = len(classes) # total number of classes
n, K

"""Implement Gibbs Sampler"""

class GibbsSampler():
    def __init__(self, X, Y, alpha, K, sigma, lam):
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.sigma = sigma
        self.lam = lam
        self.K = K
        self.theta = np.ones(K) * 1/K
        self.beta = np.zeros((K, self.X.shape[1]))
        self.Z = np.argmax(np.random.multinomial(1, self.theta, self.X.shape[0]), axis=1)
        self.normalizer = (1 / ((self.sigma ** 2) * np.sqrt(2 * np.pi)))

    def compute_log_prob(self):
        log_probs = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            # x_minus_mu = np.linalg.norm(self.X[i, :] - self.beta[self.Z[i], :])
            # log_probs[i] = (-(x_minus_mu ** 2) / (2 * (self.sigma ** 2))) + np.log(self.normalizer) + np.log(self.theta[self.Z[i]])
            log_probs[i] = np.log(self.theta[self.Z[i]]) + np.log(stats.multivariate_normal(self.beta[self.Z[i]], self.sigma).pdf(self.X[i]))

        return np.sum(log_probs)

    def sample_mixture_proportions(self):
        post_alpha = np.zeros(self.K)
        for k in range(self.K):
            k_indices = np.argwhere(self.Z == k)
            n_k = len(k_indices)
            post_alpha[k] = self.alpha[k] + n_k

        self.theta = np.random.dirichlet(post_alpha, 1)[0]

    def sample_mixture_components(self):
        for k in range(self.K):
            k_indices = np.argwhere(self.Z == k)
            n_k = len(k_indices)
            x = self.X[k_indices, :]
            x_bar = (np.sum(x, axis=0)[0] / float(n_k))

            beta_mean = ((n_k / self.sigma**2) / (n_k/self.sigma**2 + 1/self.lam**2)) * x_bar
            beta_var = 1 / (n_k/self.sigma**2 + 1/self.lam**2)
            
            self.beta[k, :] = np.random.normal(beta_mean, np.sqrt(beta_var))

    def sample_mixture_assignments(self):
        p = np.zeros((self.X.shape[0], self.K))
        for i in range(self.X.shape[0]):
            for k in range(self.K):
                x_minus_mu = np.linalg.norm(self.X[i] - self.beta[k, :])
                log_prob = (-(x_minus_mu ** 2) / (2 * (self.sigma ** 2))) + np.log(self.normalizer) + np.log(self.theta[k])
                p[i, k] = np.exp(log_prob)

            p[i, :] = p[i, :] / float(sum(p[i, :])) # normalization
            self.Z[i] = np.argmax(np.random.multinomial(n=1, pvals=p[i, :], size=1)[0])

def plot_prob(log_probs):
    iterations = np.arange(0, len(log_probs))
    plt.plot(iterations, log_probs)
    plt.title('Log Probability vs. Iterations')

    plt.savefig("log_prob.png")


if __name__ == '__main__':
    # initialize gibbs sampler
    gibbs_sampler = GibbsSampler(X, Y, alpha=np.ones(K), K=K, sigma=0.1, lam=1)

    iterations = 1000
    log_probs = []
    for iteration in range(iterations):
        gibbs_sampler.sample_mixture_proportions()
        gibbs_sampler.sample_mixture_components()
        gibbs_sampler.sample_mixture_assignments()
        log_probs.append(gibbs_sampler.compute_log_prob())
        if iteration % 50 == 0:
            print(f"Iteration [{iteration+1}] ", log_probs[-1])

    plot_prob(log_probs)

    # with open('param.npy', 'rb') as f:
    #     theta = np.load(f)
    #     beta = np.load(f)
    #     Z = np.load(f)

    # for k in range(K):
    #     bins, counts = np.unique(Y[np.argwhere(Z==k)],return_counts=True)
    #     plt.bar(bins+1, counts)

    #     plt.xticks(bins)
    #     plt.xlabel("Class")
    #     plt.ylabel("Counts")
    #     plt.title(f"Counts of each class in cluster {k+1}")
    #     plt.savefig(f"vis/cluster-{k+1}.png")
    #     plt.close()
