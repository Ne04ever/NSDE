import matplotlib.pyplot as plt
import torch

def plot_loss_curves(results: dict[str, list[float]]):

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

def make_pred(model, test_loader, ytest):
    predictions = []
    import torch.nn.functional as F
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(test_loader):
            pred = model(X)
            predictions.append(pred)

    predictions = torch.cat(predictions)  # Concatenate predictions from all batches
    mse = F.mse_loss(predictions, ytest)

    plt.plot(predictions, label='predicted')
    plt.plot(ytest, label='test')
    plt.legend()
    plt.title(f'MSE: {mse.item():.4f}')  # Display the calculated MSE in the plot title
    plt.show()
    return predictions
