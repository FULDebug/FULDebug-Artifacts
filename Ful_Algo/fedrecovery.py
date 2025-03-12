import copy
import torch
import torch.nn.functional as F
import numpy as np


class FedRecovery:
    """
    A class-based implementation of FedRecovery, with *internal* methods for evaluation
    instead of file logging or an external trainer class.
    """

    def __init__(
        self,
        device,
        old_global_model_list,
        old_local_model_list,
        ul_client,
        base_model,              # Either an nn.Module or a callable returning nn.Module
        train_loader,            # Dataloader for training set (used for evaluation)
        val_loader,              # Dataloader for validation set
        ul_loader,               # Dataloader for the "unlearn" set
        std_list=[0.02, 0.025],  # Range of noise stds to try
        num_classes=10,
        seed=0,
        num_users=10,
        batch_size=32,
        iid=True,
        lr=0.01,
        noise_scale=5.0
    ):
        """
        Parameters
        ----------
        device : torch.device
            CPU or GPU device.
        old_global_model_list : list of dict
            List of state_dicts for global models from prior FL rounds.
        old_local_model_list : list of list of dict
            List (over rounds) of lists of local model state_dicts.
        ul_client : int
            Index of the client to unlearn.
        base_model : nn.Module or callable
            If nn.Module, we copy it. If callable, we instantiate it (e.g., base_model(num_classes=...)).
        train_loader, val_loader, ul_loader : DataLoader
            Dataloaders for training data, validation data, and unlearn data (backdoor/forget set).
        std_list : list of float
            Standard deviations for the Gaussian noise to inject after each FedRecovery step.
        num_classes : int
            Number of classes in your classification task.
        seed : int
            For reproducibility/logging.
        num_users : int
            Total FL clients (for reference).
        batch_size : int
            Batch size used in training (for reference).
        iid : bool
            Whether data partition was IID (for reference).
        lr : float
            A learning rate reference (for reference).
        noise_scale : float
            Multiplier for the FedRecovery residual correction (default 5.0).
        """

        self.device = device
        self.old_global_model_list = old_global_model_list
        self.old_local_model_list = old_local_model_list
        self.ul_client = ul_client
        self.std_list = std_list
        self.num_classes = num_classes
        self.seed = seed
        self.num_users = num_users
        self.batch_size = batch_size
        self.iid = iid
        self.lr = lr
        self.noise_scale = noise_scale

        # Create (or copy) the base model
        if isinstance(base_model, torch.nn.Module):
            self.model = copy.deepcopy(base_model).to(device)
        else:
            # assume it's a factory method returning a fresh model
            self.model = base_model(num_classes=num_classes).to(device)

        # Dataloaders for evaluation
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ul_loader = ul_loader

    def ensure_state_dict_list(self, model_list):
        """Convert a list of model objects or state dicts into a uniform list of state dicts."""
        converted = []
        for item in model_list:
            if isinstance(item, torch.nn.Module):
                # It's a model, so get its state_dict
                converted.append(item.state_dict())
            elif isinstance(item, dict) or isinstance(item, torch.nn.modules.module.OrderedDict):
                # It's already a dict or OrderedDict of parameters
                converted.append(item)
            else:
                raise TypeError(f"Expected nn.Module or dict, got {type(item)}")
        return converted

    def ensure_nested_state_dict_list(self, local_model_list):
        """
        If old_local_model_list is a list (over rounds) of lists (over clients),
        convert each client model to a state dict if needed.
        """
        converted_outer = []
        for round_list in local_model_list:
            # round_list could be [client0, client1, ...]
            converted_inner = self.ensure_state_dict_list(round_list)
            converted_outer.append(converted_inner)
        return converted_outer

    def fedrecovery_operation(self, std):
        """
        Core FedRecovery step to generate `corrected_param`:
          1) Build grad_list (deltas of global models).
          2) Compute weigh_list from norms of those deltas.
          3) For each weigh_list[i], compute grad_residual and update the last global model.
          4) Add Gaussian noise with std.

        Returns
        -------
        corrected_param : dict
            A state_dict for the recovered model after unlearning & noise.
        """
        device = self.device
        old_gm = self.ensure_state_dict_list(self.old_global_model_list)
        # 2) Convert old_local_model_list similarly
        old_lm = self.ensure_nested_state_dict_list(self.old_local_model_list)

        grad_list = []
        weigh_list = []
        grad_residual = {}

        
        with torch.no_grad():
            # 1) Build delta F_i for i in [0..len(old_gm)-2]
            for i in range(len(old_gm) - 1):
                grad_dict = {}
                for name in old_gm[i].keys():
                    grad_dict[name] = old_gm[i+1][name] - old_gm[i][name]
                # Flatten
                flat_grad = torch.cat(
                    [p.reshape((-1, 1)) for p in grad_dict.values()],
                    dim=0
                ).squeeze().to(device)
                grad_list.append(flat_grad)

            # 2) weigh_list[i] = ||deltaF_(i+1)||^2 / sum_{j=0..i} ||deltaF_j||^2
            for i in range(1, len(grad_list)):
                sum_norm = torch.tensor(0.0, device=device)
                for j in range(i):
                    sum_norm += torch.sum(grad_list[j] ** 2)
                weigh_list.append(torch.sum(grad_list[i] ** 2) / sum_norm)

            # 3) Compute corrected_param from the last global model
            corrected_param = copy.deepcopy(old_gm[-1])
            for i in range(len(weigh_list)):
                for name in corrected_param.keys():
                    # grad_residual = 1/(n-1) [ (G[i+1] - G[i]) - (L[i+1][ul] - G[i]) / n ]
                    grad_residual[name] = (
                        1.0 / (len(old_lm[i+1]) - 1)
                        * (
                            (old_gm[i+1][name] - old_gm[i][name])
                            - (
                                old_lm[i+1][self.ul_client][name]
                                - old_gm[i][name]
                            ) / len(old_lm[i+1])
                        )
                    )
                    corrected_param[name] -= self.noise_scale * weigh_list[i] * grad_residual[name]

            # 4) Add Gaussian noise
            for name in corrected_param.keys():
                noise = torch.empty_like(corrected_param[name]).normal_(0, std)
                corrected_param[name] += noise

        return corrected_param

    def _evaluate_loader(self, loader):
        """
        Evaluates self.model on a loader that returns (images, labels, indexes),
        while accounting for possible extra dimensions and channel permutations.
        """
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data in loader:
                images, labels = data
                if len(images.shape) == 5:  # If shape is [batch_size, 1, 1, 28, 28]
                    images = images.squeeze(1)  # Squeeze the second dimension [batch_size, 1, 28, 28]
                # Check if the input tensor has the correct shape for CIFAR-10
                if images.shape[1] == 32:  # Indicates the channel dimension is incorrectly set as 32
                    # Permute from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                    images = images.permute(0, 3, 1, 2)

                # Move to device
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Accumulate loss
                total_loss += loss.item() * images.size(0)

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        # Compute average loss and overall accuracy
        avg_loss = total_loss / total_samples if total_samples else 0.0
        accuracy = 100.0 * total_correct / total_samples if total_samples else 0.0
        return avg_loss, accuracy



    def _evaluate_ul_loader(self, loader):
        """
        Evaluation for unlearn set: measures the fraction of samples for which
        the prediction is INCORRECT => "unlearn success".
    
        Returns (avg_loss, unlearn_success_percent).
        """
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_incorrect = 0
        total_samples = 0
    
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # If images are 5D with an extra singleton dimension (e.g. [N, 1, 1, H, W]),
                # remove the extra dimension.
                if images.dim() == 5 and images.size(1) == 1:
                    images = images.squeeze(1)  # New shape: [N, 1, H, W]
                
                # If images are 4D and in channels-last format (i.e. [N, H, W, 3]),
                # permute them to channels-first format ([N, 3, H, W]).
                elif images.dim() == 4 and images.size(-1) == 3:
                    images = images.permute(0, 3, 1, 2)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_incorrect += (predicted != labels).sum().item()
                total_samples += labels.size(0)
    
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        unlearn_success = 100.0 * total_incorrect / total_samples if total_samples > 0 else 0.0
        return avg_loss, unlearn_success


    def execute_fedrecovery(self):
        """
        Main method:
          - For each std in std_list:
            1) Run fedrecovery_operation(std)
            2) Load the recovered model
            3) Evaluate on train_loader, val_loader, ul_loader
            4) Print the results
        """
        for std in self.std_list:
            # 1) Generate new state dict
            recovered_state = self.fedrecovery_operation(std)

            # 2) Load into self.model
            self.model.load_state_dict(recovered_state)

            # 3) Evaluate on train, val, ul
            train_loss, train_acc = self._evaluate_loader(self.train_loader)
            val_loss, val_acc = self._evaluate_loader(self.val_loader)
            ul_loss, ul_effect = self._evaluate_ul_loader(self.ul_loader)

            # 4) Print results
            print(f"[FedRecovery] std={std:.3f} | "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}% | "
                  f"UL Loss={ul_loss:.4f}, UL Effect={ul_effect:.2f}%")

        # Optionally return the final recovered model
        return self.model
