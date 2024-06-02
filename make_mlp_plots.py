import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def main(X, y, t1, t2, name_file):
    # Create random input and output data
    # X = np.random.rand(10000, 1).astype(np.float32)
    # y = (X +  1*np.random.rand(10000, 1).astype(np.float32))**2

    # Convert numpy arrays to torch tensors
    X_train = torch.tensor(X)
    y_train = torch.tensor(y)

    # Define model, loss function, and optimizer
    input_size = X_train.shape[1]
    hidden_size = 100
    output_size = y_train.shape[1]

    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Testing the model
    with torch.no_grad():
        predicted = model(X_train).detach().numpy()

    # Compute residuals
    # Assuming y_train and predicted are defined and converted to numpy arrays
    residuals = y_train.numpy() - predicted

    print(len(residuals), len(residuals[residuals > 0]))

    # Calculate mean and variance
    mean_residuals = np.mean(residuals)
    variance_residuals = np.var(residuals)

    fig, ax = plt.subplots(1, 2, figsize=(9,4))
    fig.subplots_adjust(wspace=0.4)  # Increase the horizontal space (default is 0.2)

    ax1, ax2 = ax

    # Plot histogram
    ax2.hist(residuals, bins=20, histtype='step', label='Residuals', density=True)

    # Add annotations for mean and variance
    ax2.axvline(mean_residuals, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_residuals:.2f}')
    ax2.axvline(mean_residuals + np.sqrt(variance_residuals), color='g', linestyle='dotted', linewidth=1,
                label=f'Stdev: {np.sqrt(variance_residuals):.2f}')
    ax2.axvline(mean_residuals - np.sqrt(variance_residuals), color='g', linestyle='dotted', linewidth=1)

    # Add legend
    ax2.legend()

    # Display plot
    ax2.set_title('MLP Residuals')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency (a.u.)')
    # plt.show()

    ax1.set_title('Data')
    ax1.scatter(X, y, cmap='viridis', alpha=0.2, edgecolors='w', s=2)  # Use colormap and style
    ax1.set_xlabel(t1)
    ax1.set_ylabel(t2)

    if not os.path.exists('out'):
        os.mkdir('out')
    plt.savefig('out/'+name_file+'.pdf')


    # Print the first 5 predicted values, true values, and residuals
    print("First 5 Predicted values:", predicted[:5])
    print("First 5 True values:", y_train[:5].numpy())
    print("First 5 Residuals:", residuals[:5])



if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    X = np.random.normal(0, 1, (10000, 1)).astype(np.float32)
    y = (X + np.random.normal(0, 1, (10000, 1)).astype(np.float32)) ** 2
    main(X, y,'$x\sim\mathcal{N}(0,1)$', '$y\sim(x+\mathcal{N}(0,1))^2$', 'y_eq_xsq')


    X = np.linspace(0,5, num=10000).astype(np.float32)
    a = X+np.random.normal(-3, 1, (10000))
    b = X+np.random.normal(+3, 1, (10000))
    y = np.where(np.random.binomial(1, 0.5, size=10000)==0, a, b).astype(np.float32)
    # y = (X + np.random.normal(0, 1, (10000, 1)).astype(np.float32)) ** 2
    main(X[:, np.newaxis], y[:, np.newaxis],'$x\sim\mathcal{U}(0,5)$', '$y \sim x + \mathcal{N}(-3, 1) + \mathcal{N}(3, 1)$', 'y_eq_x_p_2dg')
