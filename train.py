import wandb
from barbar import Bar
from utils import binary_accuracy


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for idx, (text, label) in enumerate(Bar(iterator)):

        optimizer.zero_grad()

        predictions = model(text).squeeze(1)

        loss = criterion(predictions, label)

        acc = binary_accuracy(predictions, label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
