import torch
import wandb
from sklearn.model_selection import train_test_split
import tqdm

class Trainer:
    """ Class to perform training. """
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 validation_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 initial_learning_rate,
                 number_of_epochs,
                 report_to_wandb=False,
            ):
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.number_of_epochs = number_of_epochs
        self.report_to_wandb = report_to_wandb

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda")

        if self.report_to_wandb:
            wandb.init()

    def train(self):
        if self.use_cuda:
            self.model.to(self.device)

        for epoch_id in range(self.number_of_epochs):
            self.model.train()
            for batch in tqdm.tqdm(self.train_dataloader):
                inputs, padded_targets = batch

                if self.use_cuda:
                    inputs = inputs.to(self.device)
                    padded_targets = padded_targets.to(self.device)

                self.optimizer.zero_grad()

                # Passing the padded targets to the model, it's then used as teacher forcing in the LSTM.
                # TODO: Might want to change that.
                predictions = self.model(inputs, padded_targets)

                losses = []
                for i in range(padded_targets.shape[1]):
                    losses.append(self.loss_fn(predictions[:, i, :], padded_targets[:, i, :]))

                loss = torch.sum(torch.stack(losses))
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step(loss)

                if self.report_to_wandb:
                    wandb.log({"training_loss": loss.clone().cpu().detach().numpy()})
                    wandb.watch(self.model, criterion=self.loss_fn)
                else:
                    print("\ttraining loss: {}".format(loss.clone().cpu().detach().numpy()))

            self.model.eval()
            validation_loss = 0
            correct = 0
            for batch in self.validation_dataloader:
                inputs, targets = batch
                targets = targets.type(torch.LongTensor)

                if self.use_cuda:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                pred = outputs.data.max(1, keepdim=True)[1]
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

            validation_loss /= len(self.validation_dataloader.dataset)
            validation_accuracy = correct / len(self.validation_dataloader.dataset)

            if self.report_to_wandb:
                wandb.log({"validation_loss": validation_loss, "validation_accuracy": validation_accuracy})

            print("[Epoch #{eid}] Validation loss: {vl}".format(eid=epoch_id, vl=validation_loss))
            print("\tValidation accuracy: {va}".format(va=validation_accuracy))


def custom_collate_fn(data):
    images = []
    captions = []

    for x, y in data:
        images.append(x)
        captions.append(y)

    # Pad the target sentences
    padded_targets = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0.0)

    images_batch = torch.stack(images, dim=-1).permute(3, 0, 1, 2)
    return images_batch, padded_targets


def train_model(model, dataset, args):
    """ Trains the given model on the given dataset. """
    # Split into train and test sets.
    reduced_dataset = int(len(dataset) * args.percentage)
    end_train_index = int(0.8 * reduced_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,#[:end_train_index],
        batch_size=8,
        collate_fn=custom_collate_fn
    )
    validation_dataloader = torch.utils.data.DataLoader(
        dataset,#[end_train_index:reduced_dataset],
        batch_size=8,
        collate_fn=custom_collate_fn
    )

    trainer = Trainer(model,
                      train_dataloader,
                      validation_dataloader,
                      None,
                      0.01,
                      args.epochs,
                      report_to_wandb=False,
                      )
    trainer.train()

    return
