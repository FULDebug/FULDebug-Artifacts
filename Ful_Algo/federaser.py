import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FedEraser:
    """
    A FedEraser implementation that mimics the structure/style of the IBMFUL code.
    The main difference is in how the unlearning step is computed (unlearning_step_once).
    """

    def __init__(self,
                 client_models,
                 num_parties,
                 global_model,
                 trainloader_lst,
                 testloader,
                 testloader_poison,
                 initial_model,
                 selected_CMs,         # List of lists of client models used at each epoch (old client models)
                 selected_GMs,         # List of global models at each epoch
                 unlearn_global_models,# A list that tracks global models after each unlearning epoch
                 forget_local_epoch_ratio=0.5,
                 local_epoch=5,
                 global_epoch=5,
                 party_to_be_erased=0,
                 lr=0.01):
        """
        Constructor for FedEraser.

        Parameters
        ----------
        client_models : list of nn.Module
            Current client models, typically used to warm-start the unlearning process
            or for reference in training.
        global_model : nn.Module
            The current global model (before unlearning).
        trainloader_lst : list of DataLoader
            A list of PyTorch DataLoaders, one per client.
        testloader : DataLoader
            A DataLoader for clean test data.
        testloader_poison : DataLoader
            A DataLoader for poison/backdoor test data (if any). Can be None if not used.
        initial_model : nn.Module
            A reference model architecture (freshly initialized) if needed, 
            or the same architecture used for the global model.
        selected_CMs : list
            A list (indexed by epoch) of client model groups that were used in normal FL training 
            (i.e., old client models).
        selected_GMs : list
            A list (indexed by epoch) of global models from normal FL training.
        unlearn_global_models : list
            A list to store or track the global models produced at each unlearning epoch.
        forget_local_epoch_ratio : float
            The ratio for adjusting the local training epochs during unlearning.
        local_epoch : int
            The default local epoch used in normal FL training (will be reduced for forgetting).
        global_epoch : int
            The default number of global epochs.
        lr : float
            Learning rate for local training, if further local training is performed.
        """
        self.client_models = client_models
        self.global_model = global_model
        self.trainloader_lst = trainloader_lst
        self.testloader = testloader
        self.testloader_poison = testloader_poison
        self.initial_model = initial_model
        self.party_to_be_erased = party_to_be_erased

        # FedEraser-specific data/parameters
        self.selected_CMs = selected_CMs
        self.selected_GMs = selected_GMs
        self.unlearn_global_models = unlearn_global_models

        self.forget_local_epoch_ratio = forget_local_epoch_ratio
        self.local_epoch = local_epoch
        self.global_epoch = global_epoch
        self.lr = lr

        # Book-keeping for post-unlearning accuracy/loss
        self.global_train_acc_after_unlearn = []
        self.global_train_loss_after_unlearn = []

    def test(self, model, testloader):
        """
        Evaluate a model on a testloader that ideally yields (images, labels).
        This version attempts to handle cases where the data might be mis-specified.
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            # 1) Check if 'testloader' is actually a DataLoader.
            if isinstance(testloader, torch.utils.data.DataLoader):
                # We'll iterate over 'testloader' expecting each batch to be (images, labels, ...)
                for batch in testloader:
                    # -- A. If batch is already a tuple/list, handle typical or extra fields
                    if isinstance(batch, (tuple, list)):
                        if len(batch) == 2:
                            images, labels = batch
                        elif len(batch) >= 2:
                            images, labels = batch[0], batch[1]
                            # If there's more data (e.g. batch[2]) and you need it, handle it here
                        else:
                            # This is unusual; skip or handle
                            continue
                    else:
                        # -- B. 'batch' might be a single Tensor if you custom-coded your dataset
                        # or, in a worst-case scenario, 'batch' might be another DataLoader (!) 
                        # which indicates a double-wrapping mistake in your pipeline.
                        print("Warning: 'batch' is not a tuple/list. Check your data pipeline.")
                        continue

                    # -- C. If shape is 5D or channels are in the wrong dimension, fix as needed
                    if len(images.shape) == 5:  # e.g. [batch_size, 1, 1, H, W]
                        images = images.squeeze(1)  # from [N,1,H,W] to [N,H,W]
                    if images.ndim == 4 and images.shape[1] == 32:
                        # Possibly CIFAR-like data but channels last. Permute from [N, H, W, C] -> [N, C, H, W]
                        images = images.permute(0, 3, 1, 2)

                    # -- D. Compute loss & accuracy
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * images.size(0)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            else:
                # 2) If 'testloader' isn't even a DataLoader, you might be dealing
                #    with a single (images, labels) batch or something else.
                print("Warning: 'testloader' is not a DataLoader. Please check usage.")
                # Here, you could handle a single batch scenario if needed.
                # For example:
                # if isinstance(testloader, (tuple, list)) and len(testloader) == 2:
                #     images, labels = testloader
                #     # do the same evaluation steps
                # else:
                #     return 0.0, 0.0
                return 0.0, 0.0

        # 3) Compute average loss and accuracy
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return avg_loss, accuracy

        

    def global_train_once(self, global_model, client_data_loaders):
        """
        Example method: performs one round of local training on each client
        and returns the updated client models. In practice, you might do multiple 
        local epochs and then average. This is just a skeleton.

        Parameters
        ----------
        global_model : nn.Module
            The current global model to be distributed to all clients.
        client_data_loaders : list of DataLoader
            The local dataloaders for each client.

        Returns
        -------
        new_client_models : list of nn.Module
            Updated client models after one round of local training.
        """
        new_client_models = []
        for i, loader in enumerate(client_data_loaders):
            if i == self.party_to_be_erased:
                continue
            local_model = copy.deepcopy(global_model)
            optimizer = optim.SGD(local_model.parameters(), lr=self.lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            local_model.train()
            for _ in range(int(self.local_epoch)):  # local epochs
                for images, labels in loader:
                    if len(images.shape) == 5:  # If shape is [batch_size, 1, 1, 28, 28]
                        images = images.squeeze(1)  # Squeeze the second dimension [batch_size, 1, 28, 28]
                    # Check if the input tensor has the correct shape for CIFAR-10
                    if images.shape[1] == 32:  # Indicates the channel dimension is incorrectly set as 32
                        # Permute from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                        images = images.permute(0, 3, 1, 2)
                    optimizer.zero_grad()
                    outputs = local_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            new_client_models.append(local_model)
        return new_client_models

    def omit_party_in_selected_CMs(self, selected_CMs):
        selected_CMs = [
            model
            for i, model in enumerate(selected_CMs)
            if i != self.party_to_be_erased
        ]
        return selected_CMs

    def unlearning_step_once(self, old_client_models, new_client_models, global_model_before_forget, global_model_after_forget):
        """
        Core FedEraser update rule for unlearning.

        old_client_models : list of nn.Module
            The older set of client models (from normal FL training) 
            that do NOT include the forgotten user(s).
        new_client_models : list of nn.Module
            Newly trained client models (using fewer epochs / forgetting settings).
        global_model_before_forget : nn.Module
            The global model before forgetting (old global).
        global_model_after_forget : nn.Module
            The newly updated global model (post local training with forgetting).

        Returns
        -------
        return_global_model : nn.Module
            The updated global model after a single unlearning iteration.
        """
        # We will combine the states as described in the snippet:
        #   return_model_state[layer] = new_global_state[layer] 
        #       + ||oldCM - oldGM|| * ( (newCM - newGM) / ||newCM - newGM|| )

        old_param_update = {}
        new_param_update = {}
        return_model_state = {}

        # New global model's state dict
        new_global_model_state = global_model_after_forget.state_dict()

        # For convenience, get the state dicts of "before" and "after" forgetting
        gmbf_state = global_model_before_forget.state_dict()  # oldGM
        gmaf_state = global_model_after_forget.state_dict()   # newGM

        # We assume old_client_models and new_client_models have the same length
        # (the set of active/remaining clients).
        assert len(old_client_models) == len(new_client_models), \
            "Mismatch in number of old and new client models."

        for layer in gmbf_state.keys():
            # Initialize
            old_param_update[layer] = 0 * gmbf_state[layer]
            new_param_update[layer] = 0 * gmbf_state[layer]
            return_model_state[layer] = 0 * gmbf_state[layer]

            # Sum across all client models
            for i in range(len(new_client_models)):
                old_param_update[layer] += old_client_models[i].state_dict()[layer]
                new_param_update[layer] += new_client_models[i].state_dict()[layer]

            # Average across clients
            old_param_update[layer] /= (len(new_client_models))
            new_param_update[layer] /= (len(new_client_models))

            # old_param_update[layer] = oldCM - oldGM
            old_param_update[layer] -= gmbf_state[layer]

            # new_param_update[layer] = newCM - newGM
            new_param_update[layer] -= gmaf_state[layer]

            # Step length = ||oldCM - oldGM||
            step_length = torch.norm(old_param_update[layer])

            # Step direction = (newCM - newGM) / ||newCM - newGM||
            norm_new_update = torch.norm(new_param_update[layer])
            if norm_new_update > 0:
                step_direction = new_param_update[layer] / norm_new_update
            else:
                step_direction = 0 * new_param_update[layer]

            # return_model_state = new_global_model + step_length * step_direction
            return_model_state[layer] = new_global_model_state[layer] + step_length * step_direction

        # Build the final global model
        return_global_model = copy.deepcopy(global_model_after_forget)
        return_global_model.load_state_dict(return_model_state)

        return return_global_model

    def execute_unlearning(self):
        """
        Run the FedEraser unlearning process. This mimics the snippet logic:
          1. Temporarily adjust local epoch to be local_epoch * forget_local_epoch_ratio
          2. Perform a series of global epochs (the snippet used CM_intv.shape[0]; adapt as needed)
          3. Each global epoch:
             - Evaluate poison performance (optional)
             - Do local training to get new client models
             - Perform unlearning_step_once to get a new global model
             - Evaluate on test set
             - Append the updated global model
        4. Restore local_epoch / global_epoch to their original values
        5. Return the list of unlearned global models
        """
        # 1. Cache the original local/global epochs
        CONST_local_epoch = copy.deepcopy(self.local_epoch)
        CONST_global_epoch = copy.deepcopy(self.global_epoch)

        # 2. Adjust local_epoch based on forget_local_epoch_ratio
        self.local_epoch = int(np.ceil(self.local_epoch * self.forget_local_epoch_ratio))
        # In the snippet, global_epoch was redefined based on some interval shape. 
        # We'll assume the length of 'selected_CMs' indicates the unlearning epochs:

        print(f"[FedEraser] Adjusted local_epoch for forgetting = {self.local_epoch}")
        print(f"[FedEraser] Unlearning will run for {self.global_epoch} global epochs")
        # 3. Unlearning loop
        for epoch in range(self.global_epoch):
            if epoch == 0:
                # According to snippet, skip or continue (since we might already have an initial model).
                continue

            print(f"------ FedEraser Global Epoch = {epoch} ------")

            # The snippet: unlearn_global_models[epoch] is the reference global model for this step
            current_global_model = self.unlearn_global_models[epoch]

            # Evaluate on poison set (optional)
            if self.testloader_poison is not None:
                initial_model = self.initial_model
                current_global_model = copy.deepcopy(initial_model)
                current_global_model.load_state_dict(self.unlearn_global_models[epoch])
                pois_loss, pois_acc = self.test(current_global_model, self.testloader_poison)
                print(f"Poison accuracy before unlearning step = {pois_acc:.2f}%")

            # Perform local training with the *reduced local_epoch*, get new client models
            new_client_models = self.global_train_once(current_global_model, self.trainloader_lst)

            # The snippet uses: 
            #   new_GM = unlearning_step_once(selected_CMs[epoch], new_client_models, selected_GMs[epoch+1], current_global_model)
            #   But be mindful of index bounds (epoch+1 might need to be within selected_GMs).
            initial_model = self.initial_model
            global_model_before_forget = copy.deepcopy(initial_model)
                
            if (epoch + 1) < len(self.selected_GMs):
                global_model_before_forget.load_state_dict(self.selected_GMs[epoch + 1])
            else:
                # If for some reason we are at the last epoch and there's no GM[epoch+1],
                # fallback to the last known global model or handle index carefully.
                global_model_before_forget.load_state_dict(self.selected_GMs[epoch + 1])

            new_global_model = self.unlearning_step_once(
                old_client_models=self.omit_party_in_selected_CMs(self.selected_CMs[epoch]),
                new_client_models=new_client_models,
                global_model_before_forget=global_model_before_forget,
                global_model_after_forget=current_global_model
            )

            # Evaluate the new global model on the clean test set
            test_loss, test_acc = self.test(new_global_model, self.testloader)
            print(f"Clean test accuracy after unlearning step = {test_acc:.2f}%")

            self.global_train_acc_after_unlearn.append(test_acc)
            self.global_train_loss_after_unlearn.append(test_loss)

            # Append the new global model to the unlearn_global_models
            self.unlearn_global_models.append(new_global_model)

        # 4. Restore original epochs
        self.local_epoch = CONST_local_epoch
        self.global_epoch = CONST_global_epoch

        # 5. Return the updated list of unlearned global models
        return self.unlearn_global_models
