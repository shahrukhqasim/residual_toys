import os.path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Number of samples




# Define the encoder network
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc1_3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc1_2(x))
        h = torch.relu(self.fc1_3(x))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar

# Define the decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.fc1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc1_3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, y):
        x = torch.relu(self.fc1(torch.cat((z, y), dim=1)))
        x = torch.relu(self.fc1_2(x))
        h = torch.relu(self.fc1_3(x))
        out = self.fc2(h)
        return out

# Define the VAE
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super(ConditionalVAE, self).__init__()
        self.encoder = Encoder(input_dim + condition_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, condition_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        input_data = torch.cat((x, y), dim=1)
        mu, logvar = self.encoder(input_data)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z, y)
        return out, mu, logvar



n_samples = 1000


# # Generate X ~ N(0,1)
# X = np.random.normal(0, 10, n_samples)
# # Generate Y = (X + N(0,1))**2
# Y = (X + np.random.normal(0, 1, n_samples))



def main(sample_fn, t1, t2, filename):

    # Hyperparameters
    input_dim = 1
    condition_dim = 1
    hidden_dim = 512
    latent_dim = 1
    learning_rate = 0.0001
    num_epochs = 10000
    kld_weight = 0.0025

    # Prepare the data
    # X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    # Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    # dataset = TensorDataset(X_tensor, Y_tensor)
    # dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize the model, optimizer and loss function
    vae = ConditionalVAE(input_dim, condition_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    reconstruction_loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        # for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        X_tensor, Y_tensor = sample_fn()
        recon_y, mu, logvar = vae(Y_tensor, X_tensor)
        recon_loss = reconstruction_loss_fn(recon_y, Y_tensor)
        # kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        loss = recon_loss + kld_weight * kld_loss
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training completed!")


    # Generate new samples from the trained VAE conditioned on new Y values

    new_X, y_check = sample_fn()

    # new_X_tensor = torch.tensor(new_X, dtype=torch.float32)


    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        generated_Y = vae.decoder(z, new_X).numpy()

    # fig, ax = plt.subplots(2, 2, figsize=(9, 8))
    # fig.subplots_adjust(wspace=0.4, hspace=0.3)  # Increase the horizontal space (default is 0.2)
    # (ax1, ax2), (ax3, ax4) = ax

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    fig.subplots_adjust(wspace=0.4)  # Increase the horizontal space (default is 0.2)
    ax1, ax2 = ax


    X, Y = sample_fn()
    X = X.numpy().flatten()
    Y = Y.numpy().flatten()
    ax1.scatter(X, Y, alpha=0.5, s=0.1)
    ax1.set_xlabel(t1)
    ax1.set_ylabel(t2)
    ax1.set_title('Given data')
    # plt.show()

    # Plot the learned joint distribution of X and Y
    # plt.figure(figsize=(8, 6))
    # ax2.scatter(new_X, y_check, alpha=0.3, s=0.1,color='blue')
    ax2.scatter(new_X, generated_Y, alpha=0.5, s=0.1,color='red')
    ax2.set_xlabel(t1)
    ax2.set_ylabel(t2)
    ax2.set_title('VAE sampled')

    if not os.path.exists('out'):
        os.mkdir('out')
    fig.savefig('out/'+filename+'.pdf')


    # # Compute residuals
    # residuals = - generated_Y.flatten() + y_check.numpy().flatten()
    #
    # print(len(residuals), len(residuals[residuals>0]))
    #
    # # Calculate mean and variance
    # mean_residuals = np.mean(residuals)
    # variance_residuals = np.var(residuals)
    #
    # # Plot histogram
    # ax3.hist(residuals, bins=20, histtype='step', label='Residuals')
    #
    # # Add annotations for mean and variance
    # ax3.axvline(mean_residuals, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_residuals:.2f}')
    # ax3.axvline(mean_residuals + np.sqrt(variance_residuals), color='g', linestyle='dotted', linewidth=1, label=f'Stdev: {np.sqrt(variance_residuals):.2f}')
    # ax3.axvline(mean_residuals - np.sqrt(variance_residuals), color='g', linestyle='dotted', linewidth=1)
    #
    # # Add legend
    # ax3.legend()
    #
    # # Display plot
    # ax3.set_title('Histogram of Residuals')
    # ax3.set_xlabel('Residuals')
    # ax3.set_ylabel('Frequency')
    # plt.show()


    # n_samples_2 = 10000
    # new_X = np.random.normal(0, 10, size=(n_samples_2,1),)*0.0
    # new_X_tensor = torch.tensor(new_X, dtype=torch.float32)
    #
    # with torch.no_grad():
    #     z = torch.randn(n_samples_2, latent_dim)
    #     generated_Y = vae.decoder(z, new_X_tensor).numpy()
    #
    # # plt.hist(generated_Y.flatten(), histtype='step', bins=20)
    #
    #
    # # Calculate mean and variance
    # mean_gen = np.mean(generated_Y.flatten())
    # var_gen = np.var(generated_Y.flatten())
    #
    # # Plot histogram
    # plt.hist(generated_Y.flatten(), bins=20, histtype='step', label='Gen y | x = 0')
    #
    # # Add annotations for mean and variance
    # plt.axvline(mean_gen, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_residuals:.2f}')
    # plt.axvline(mean_gen + np.sqrt(var_gen), color='g', linestyle='dotted', linewidth=1, label=f'Stdev: {np.sqrt(variance_residuals):.2f}')
    # plt.axvline(mean_gen - np.sqrt(var_gen), color='g', linestyle='dotted', linewidth=1)
    #
    # plt.legend()

    # plt.show()


def sample_x_y_eq_xq():
    # Generate X ~ N(0,1)
    X = np.random.normal(0, 1, n_samples)
    # Generate Y = (X + N(0,1))**2
    Y = (X + np.random.normal(0, 1, n_samples))**2

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

    return X_tensor, Y_tensor

def sample_gmm():
    X = np.linspace(0,5, num=n_samples).astype(np.float32)
    a = X+np.random.normal(-3, 1, (n_samples))
    b = X+np.random.normal(+3, 1, (n_samples))
    y = np.where(np.random.binomial(1, 0.5, size=n_samples)==0, a, b).astype(np.float32)

    return torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.float32).unsqueeze(1)


def sample_simple():
    # Generate X ~ N(0,1)
    X = np.random.normal(0, 1, n_samples)
    # Generate Y = (X + N(0,1))**2
    Y = (X**2 + np.random.normal(0, 0.3, n_samples))

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

    return X_tensor, Y_tensor

if __name__ == '__main__':

    main(sample_simple, '$x\sim\mathcal{N}(0,1)$', '$y\sim(x^2+\mathcal{N}(0,1))$', 'vae_y_eq_xsq_noise')
    main(sample_x_y_eq_xq, '$x\sim\mathcal{N}(0,1)$', '$y\sim(x+\mathcal{N}(0,1))^2$', 'vae_y_eq_xsq')
    main(sample_gmm, '$x\sim\mathcal{U}(0,5)$', '$y \sim x + \mathcal{N}(-3, 1) + \mathcal{N}(3, 1)$','vae_y_eq_x_p_2dg')