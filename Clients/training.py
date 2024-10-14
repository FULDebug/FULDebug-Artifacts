import torch
from torch import nn

class Training:

    """
    Base class for Local Training
    """

    def __init__(self, 
                 num_updates_in_epoch=None,
                 num_local_epochs=1):
       
        self.name = "training"
        self.num_updates = num_updates_in_epoch
        self.num_local_epochs = num_local_epochs
        

    def train(self, model, trainloader, criterion=None, opt=None, lr = 1e-2, dataType="MNIST"):
        
        """
        Method for local training
        """
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if opt is None:
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        if self.num_updates is not None:
            self.num_local_epochs = 1

        model.train()
        running_loss = 0.0
        for epoch in range(self.num_local_epochs):
            for batch_id, (data, target) in enumerate(trainloader):
                x_batch, y_batch = data, target
                # Check if the tensor has 5 dimensions and squeeze it if needed
                # Ensure that input and target batch sizes match
                
                # Check initial batch sizes
                # print(f"Initial input batch size: {x_batch.shape[0]}, Target batch size: {y_batch.shape[0]}")

                if x_batch.dim() == 5:
                    x_batch = x_batch.squeeze(1)  # Removes the extra dimension if it has size 1

                # Rearrange the dimensions of CIFAR-10 images to [batch_size, channels, height, width]
                if dataType == "CIFAR":
                    # Check if the input tensor has the correct shape for CIFAR-10
                    if x_batch.shape[1] == 32:  # Indicates the channel dimension is incorrectly set as 32
                        # Permute from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                        x_batch = x_batch.permute(0, 3, 1, 2)

                    # x_batch = x_batch.permute(0, 3, 1, 2)  # [batch_size, channels, height, width]

                opt.zero_grad()

                outputs = model(x_batch)
                # Log the output and target shape
                # print(f"Model output shape: {outputs.shape}, Target shape: {y_batch.shape}")

                loss = criterion(outputs, y_batch)

                loss.backward()
                opt.step()
                
                running_loss += loss.item()

                if self.num_updates is not None and batch_id >= self.num_updates:
                    break

        return model, running_loss/(batch_id+1)
        
        
    def evaluate(self, testloader, model):

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # Ensure that the images are of shape [batch_size, channels, height, width]
                if len(images.shape) == 5:  # If shape is [batch_size, 1, 1, 28, 28]
                    images = images.squeeze(1)  # Squeeze the second dimension [batch_size, 1, 28, 28]
                
                
                # Check if the input tensor has the correct shape for CIFAR-10
                if images.shape[1] == 32:  # Indicates the channel dimension is incorrectly set as 32
                    # Permute from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                    images = images.permute(0, 3, 1, 2)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total
