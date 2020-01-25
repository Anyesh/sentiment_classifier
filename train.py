import wandb
from barbar import Bar
from utils import binary_accuracy


def train(model, iterator, optimizer, criterion):
    wandb.init()
    wandb.watch(model)

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in Bar(iterator):

        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
