import os
import numpy as np
import requests
import shutil
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
import tarfile
from torchvision import datasets

class DataManager:
    def __init__(self, download_dir="", normalize=True, num_clients=10):
        """
        Initializes the DataManager with the given parameters.

        :param download_dir: Directory to download data
        :type download_dir: str
        :param normalize: Whether or not to normalize data
        :type normalize: bool
        """
        self.download_dir = download_dir
        self.normalize = normalize
        self.num_clients = num_clients

    def save_file(self, path, url):
        """
        Saves a file from URL to the specified path.

        :param path: The path to save the file
        :type path: str
        :param url: The link to download from
        :type url: str
        """
        with requests.get(url, stream=True, verify=False) as r:
            with open(path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

    def load_mnist(self):
        """
        Download MNIST training data from the source used in `keras.datasets.load_mnist`

        :return: 2 tuples containing training and testing data respectively
        :rtype: (`np.ndarray`, `np.ndarray`), (`np.ndarray`, `np.ndarray`)
        """
        local_file = os.path.join(self.download_dir, "mnist.npz")
        if not os.path.isfile(local_file):
            self.save_file(local_file, "https://s3.amazonaws.com/img-datasets/mnist.npz")

            with np.load(local_file, allow_pickle=True) as mnist:
                x_train, y_train = mnist['x_train'], mnist['y_train']
                x_test, y_test = mnist['x_test'], mnist['y_test']
                if self.normalize:
                    x_train = x_train.astype('float32')
                    x_test = x_test.astype('float32')

                    x_train /= 255
                    x_test /= 255

            # Save the normalized mnist.npz
            np.savez(local_file, x_train=x_train, y_train=y_train,
                     x_test=x_test, y_test=y_test)
        else:
            with np.load(local_file, allow_pickle=True) as mnist:
                x_train, y_train = mnist['x_train'], mnist['y_train']
                x_test, y_test = mnist['x_test'], mnist['y_test']

        return (x_train, y_train), (x_test, y_test)

    def load_fashion_mnist(self):
        """
        Download Fashion-MNIST training data using torchvision.datasets.FashionMNIST.

        :return: 2 tuples containing training and testing data respectively
        :rtype: (`np.ndarray`, `np.ndarray`), (`np.ndarray`, `np.ndarray`)
        """

        local_file = os.path.join(self.download_dir, "fashion-mnist.npz")
        if not os.path.isfile(local_file):
            # Ensure the download directory exists
            os.makedirs(self.download_dir, exist_ok=True)

            # Load the datasets without any transformations to get PIL images
            train_dataset = datasets.FashionMNIST(root=self.download_dir, train=True,
                                                download=True, transform=None)
            test_dataset = datasets.FashionMNIST(root=self.download_dir, train=False,
                                                download=True, transform=None)

            # Convert datasets to numpy arrays
            x_train = np.array([np.array(img, dtype=np.float32) for img, _ in train_dataset])
            y_train = np.array([label for _, label in train_dataset], dtype=np.int64)

            x_test = np.array([np.array(img, dtype=np.float32) for img, _ in test_dataset])
            y_test = np.array([label for _, label in test_dataset], dtype=np.int64)

            # Normalize the data if required
            if self.normalize:
                x_train /= 255.0
                x_test /= 255.0

            # Add a channel dimension to the images (batch_size, 1, 28, 28)
            x_train = x_train[:, np.newaxis, :, :]
            x_test = x_test[:, np.newaxis, :, :]

            # Save the data to a local npz file
            np.savez(local_file, x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test)
        else:
            with np.load(local_file, allow_pickle=True) as fmnist:
                x_train = fmnist['x_train']
                y_train = fmnist['y_train']
                x_test = fmnist['x_test']
                y_test = fmnist['y_test']

        return (x_train, y_train), (x_test, y_test)


    def load_cifar10(self):

        """
        Download CIFAR-10 training data from the source used in `keras.datasets.load_cifar10`

        :return: 2 tuples containing training and testing data respectively
        :rtype: (`np.ndarray`, `np.ndarray`), (`np.ndarray`, `np.ndarray`)
        """

        local_file = os.path.join(self.download_dir, "cifar10.npz")
        if not os.path.isfile(local_file):
            # Download CIFAR-10 dataset
            tarpath = os.path.join(self.download_dir, "cifar-10-python.tar.gz")
            if not os.path.isfile(tarpath):
                self.save_file(tarpath, "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
            # Extract the tar.gz file
            with tarfile.open(tarpath, 'r:gz') as tar:
                tar.extractall(path=self.download_dir)
            
            # Load training data
            x_train = []
            y_train = []
            for i in range(1, 6):
                batch_file = os.path.join(self.download_dir, 'cifar-10-batches-py', f'data_batch_{i}')
                with open(batch_file, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    x_train.append(batch[b'data'])
                    y_train.extend(batch[b'labels'])
            
            # Convert training data to NumPy array and reshape to [batch_size, 32, 32, 3]
            x_train = np.concatenate(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            y_train = np.array(y_train)

            # Load testing data
            test_file = os.path.join(self.download_dir, 'cifar-10-batches-py', 'test_batch')
            with open(test_file, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                x_test = batch[b'data']
                y_test = np.array(batch[b'labels'])

            # Convert testing data to NumPy array and reshape to [batch_size, 32, 32, 3]
            x_test = np.array(x_test).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

            # Normalize data if required
            if self.normalize:
                x_train = x_train.astype('float32') / 255.0
                x_test = x_test.astype('float32') / 255.0

            # Save the dataset
            np.savez(local_file, x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test)

        else:
            with np.load(local_file, allow_pickle=True) as cifar10:
                x_train = cifar10['x_train']
                y_train = cifar10['y_train']
                x_test = cifar10['x_test']
                y_test = cifar10['y_test']

        return (x_train, y_train), (x_test, y_test)


    def split_data(self, x_train, y_train, batch_size=128):
        """
        Splits the data into the number of clients and returns DataLoaders.

        :param x_train: Training data
        :type x_train: np.ndarray
        :param y_train: Training labels
        :type y_train: np.ndarray
        :param batch_size: Batch size for the DataLoaders
        :type batch_size: int
        :return: A list of DataLoaders for each client
        :rtype: list[DataLoader]
        """
        data_loaders = []

        for i in range(self.num_clients):
            start = int(i * len(x_train) / self.num_clients)
            end = int((i + 1) * len(x_train) / self.num_clients)
            
            x_part = np.expand_dims(x_train[start:end], axis=1)
            y_part = y_train[start:end]
            
            # Create TensorDataset
            dataset = TensorDataset(torch.Tensor(x_part), torch.Tensor(y_part).long())
            
            # Create DataLoader
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            data_loaders.append(loader)
        
        return data_loaders


    def split_data_label_flipping(self, x_train, y_train, batch_size=128, num_classes=10):
        """
        Randomly splits the entire dataset among clients. For client 0, flips 75% of the labels (acts as a poisoned client).
        The remaining clients' data is correct.

        :param x_train: Training data
        :type x_train: np.ndarray
        :param y_train: Training labels
        :type y_train: np.ndarray
        :param batch_size: Batch size for the DataLoaders
        :type batch_size: int
        :param num_classes: Number of classes in the dataset
        :type num_classes: int
        :return: A list of DataLoaders for each client
        :rtype: list[DataLoader]
        """
        data_loaders = []

        # Shuffle the indices of the entire dataset
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)

        # Split the dataset randomly among all clients
        data_per_client = len(indices) // self.num_clients

        for i in range(self.num_clients):
            start_idx = i * data_per_client
            end_idx = start_idx + data_per_client
            
            x_part = np.expand_dims(x_train[indices[start_idx:end_idx]], axis=1)
            y_part = y_train[indices[start_idx:end_idx]]

            if i == 0:
                # For client 0, flip 75% of the data labels (label poisoning)
                num_samples_client_0 = len(y_part)
                num_flipped = int(0.85 * num_samples_client_0)  # 75% of client 0's data
                
                # Shuffle the data for client 0 to select random samples for flipping
                shuffled_indices = np.arange(num_samples_client_0)
                np.random.shuffle(shuffled_indices)

                # Split indices into flipped and unflipped groups
                flip_indices = shuffled_indices[:num_flipped]
                unflipped_indices = shuffled_indices[num_flipped:]

                # Flip labels for the selected 75% portion
                y_part_flipped = y_part.copy()
                y_part_flipped[flip_indices] = np.array([np.random.choice([j for j in range(num_classes) if j != label]) for label in y_part[flip_indices]])

                # Combine flipped and unflipped labels
                y_part = y_part_flipped  # 75% flipped, 25% unflipped
            
            # Create a TensorDataset and DataLoader
            dataset = TensorDataset(torch.Tensor(x_part), torch.Tensor(y_part).long())
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            data_loaders.append(loader)

        return data_loaders



    def split_data_uneven(self, x_train, y_train, batch_size=128):

        data_loaders = []

        # Separate class 1 data and non-class 1 data
        class_1_indices = np.where(y_train == 1)[0]
        non_class_1_indices = np.where(y_train != 1)[0]

        # Shuffle class 1 indices
        np.random.shuffle(class_1_indices)

        # Split 80% of class 1 data to client 0
        split_idx_80 = int(0.8 * len(class_1_indices))
        class_1_indices_client_0 = class_1_indices[:split_idx_80]
        class_1_indices_other = class_1_indices[split_idx_80:]

        # Further split the remaining 20% of class 1 data (10% each for client 1 and client 2)
        split_idx_10 = len(class_1_indices_other) // 2
        class_1_indices_client_1 = class_1_indices_other[:split_idx_10]
        class_1_indices_client_2 = class_1_indices_other[split_idx_10:]

        # Assign 80% of class 1 data to client 0
        x_class_1_client_0 = x_train[class_1_indices_client_0]
        y_class_1_client_0 = y_train[class_1_indices_client_0]

        # Create a TensorDataset for client 0 (80% of class 1 data)
        x_part_client_0 = np.expand_dims(x_class_1_client_0, axis=1)
        y_part_client_0 = y_class_1_client_0
        dataset_client_0 = TensorDataset(torch.Tensor(x_part_client_0), torch.Tensor(y_part_client_0).long())
        loader_client_0 = DataLoader(dataset_client_0, batch_size=batch_size, shuffle=True)
        data_loaders.append(loader_client_0)

        # Assign 10% of class 1 data to client 1
        x_class_1_client_1 = x_train[class_1_indices_client_1]
        y_class_1_client_1 = y_train[class_1_indices_client_1]
        x_part_client_1 = np.expand_dims(x_class_1_client_1, axis=1)
        y_part_client_1 = y_class_1_client_1
        dataset_client_1 = TensorDataset(torch.Tensor(x_part_client_1), torch.Tensor(y_part_client_1).long())
        loader_client_1 = DataLoader(dataset_client_1, batch_size=batch_size, shuffle=True)
        data_loaders.append(loader_client_1)

        # Assign 10% of class 1 data to client 2
        x_class_1_client_2 = x_train[class_1_indices_client_2]
        y_class_1_client_2 = y_train[class_1_indices_client_2]
        x_part_client_2 = np.expand_dims(x_class_1_client_2, axis=1)
        y_part_client_2 = y_class_1_client_2
        dataset_client_2 = TensorDataset(torch.Tensor(x_part_client_2), torch.Tensor(y_part_client_2).long())
        loader_client_2 = DataLoader(dataset_client_2, batch_size=batch_size, shuffle=True)
        data_loaders.append(loader_client_2)

        # Combine non-class 1 data and shuffle it
        np.random.shuffle(non_class_1_indices)

        # Split remaining non-class 1 data equally among all clients
        num_clients = self.num_clients
        data_per_client = len(non_class_1_indices) // num_clients
        
        for i in range(num_clients):
            start_idx = i * data_per_client
            end_idx = start_idx + data_per_client

            # Get non-class 1 data for each client
            x_part = np.expand_dims(x_train[non_class_1_indices[start_idx:end_idx]], axis=1)
            y_part = y_train[non_class_1_indices[start_idx:end_idx]]
            
            # Create TensorDataset and DataLoader
            dataset = TensorDataset(torch.Tensor(x_part), torch.Tensor(y_part).long())
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            if i < len(data_loaders):  # Append to existing client (client 0, 1, 2)
                data_loaders[i] = DataLoader(TensorDataset(
                    torch.cat([data_loaders[i].dataset.tensors[0], torch.Tensor(x_part)]),
                    torch.cat([data_loaders[i].dataset.tensors[1], torch.Tensor(y_part).long()])
                ), batch_size=batch_size, shuffle=True)
            else:
                data_loaders.append(loader)
        
        return data_loaders
    

    def split_data_dirichlet(self, x_train, y_train, alpha=0.1, batch_size=128):
        """
        Splits (x_train, y_train) among self.num_clients using a Dirichlet distribution.
    
        :param x_train: Numpy array of training samples, shape (num_samples, H, W) or (num_samples, H, W, C)
        :param y_train: Numpy array of labels, shape (num_samples,)
        :param alpha: Dirichlet concentration parameter. Lower alpha -> more skewed distributions.
        :param batch_size: Batch size for each client's DataLoader.
        :return: A list of DataLoaders (one for each client).
        """
    
        num_clients = self.num_clients
        data_loaders = []
    
        # Identify the unique classes in the dataset
        unique_labels = np.unique(y_train)
        
        # Prepare a list to accumulate indices for each client
        client_indices = [[] for _ in range(num_clients)]
    
        # For each class, partition its samples among the clients according to Dirichlet
        for c in unique_labels:
            class_indices = np.where(y_train == c)[0]
            np.random.shuffle(class_indices)
    
            # Draw proportions for each client from a Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * num_clients)
    
            # Round the number of samples per client
            # e.g., if proportions[i] = 0.2 and len(class_indices)=100, then client i gets ~20 samples of class c
            proportions = (proportions * len(class_indices)).astype(int)
    
            # If rounding caused a mismatch (due to rounding), fix it by adding or removing samples from the largest proportion
            total_assigned = np.sum(proportions)
            diff = len(class_indices) - total_assigned
            if diff > 0:
                # add the missing samples to the client with the largest proportion
                idx_max = np.argmax(proportions)
                proportions[idx_max] += diff
            elif diff < 0:
                # remove extra samples from the client with the largest proportion
                idx_max = np.argmax(proportions)
                proportions[idx_max] += diff
    
            # Partition the class_indices according to these proportions
            start = 0
            for i in range(num_clients):
                num_samples = proportions[i]
                client_indices[i].extend(class_indices[start:start+num_samples])
                start += num_samples
    
        # Create a DataLoader for each client
        for i in range(num_clients):
            # Extract the data for client i
            c_indices = client_indices[i]
    
            # If x_train has shape (N, H, W), expand dims to (N, 1, H, W) for PyTorch CNNs
            x_part = x_train[c_indices]
            if x_part.ndim == 3:  # (N, H, W)
                x_part = np.expand_dims(x_part, axis=1)  # -> (N, 1, H, W)
            elif x_part.ndim == 4:
                # Already has a channels dimension, e.g., (N, H, W, C).
                # If channels is last, you might need to permute to (N, C, H, W).
                # For example: x_part = np.transpose(x_part, (0, 3, 1, 2))
                pass
    
            y_part = y_train[c_indices]
    
            # Create a TensorDataset
            dataset = TensorDataset(torch.tensor(x_part, dtype=torch.float32),
                                    torch.tensor(y_part, dtype=torch.long))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            data_loaders.append(loader)
    
        return data_loaders


    def split_data_shards_rare_classes(self, x_train, y_train, rare_classes=[8,9], batch_size=128):
        """
        Splits data into shards. Rare classes are assigned to a designated client,
        while the rest of the data is split into shards and distributed among the other clients.
        
        :param x_train: NumPy array of training samples.
        :param y_train: NumPy array of training labels.
        :param rare_classes: List of labels considered rare.
        :param num_clients: Total number of clients.
        :param batch_size: Batch size for DataLoaders.
        :return: List of DataLoaders (one per client).
        """
        data_loaders = []
        num_clients = self.num_clients
        # Get indices for rare classes and non-rare classes.
        rare_idx = np.where(np.isin(y_train, rare_classes))[0]
        non_rare_idx = np.where(~np.isin(y_train, rare_classes))[0]
        
        np.random.shuffle(rare_idx)
        np.random.shuffle(non_rare_idx)
        
        # Designate client 0 to hold all rare classes.
        x_rare = x_train[rare_idx]
        y_rare = y_train[rare_idx]
        # Expand dims if necessary (for CNN input)
        if x_rare.ndim == 3:
            x_rare = np.expand_dims(x_rare, axis=1)
        dataset_rare = TensorDataset(torch.tensor(x_rare, dtype=torch.float32),
                                     torch.tensor(y_rare, dtype=torch.long))
        loader_rare = DataLoader(dataset_rare, batch_size=batch_size, shuffle=True)
        data_loaders.append(loader_rare)
        
        # The remaining clients (client 1 to num_clients-1) get shards of the non-rare data.
        # For simplicity, split non-rare data evenly among these clients.
        num_other_clients = num_clients - 1
        shard_size = len(non_rare_idx) // num_other_clients
        for i in range(num_other_clients):
            start = i * shard_size
            end = start + shard_size if i < num_other_clients - 1 else len(non_rare_idx)
            client_idx = non_rare_idx[start:end]
            x_client = x_train[client_idx]
            y_client = y_train[client_idx]
            if x_client.ndim == 3:
                x_client = np.expand_dims(x_client, axis=1)
            dataset_client = TensorDataset(torch.tensor(x_client, dtype=torch.float32),
                                           torch.tensor(y_client, dtype=torch.long))
            loader_client = DataLoader(dataset_client, batch_size=batch_size, shuffle=True)
            data_loaders.append(loader_client)
            
        return data_loaders


    def split_data_extreme_minority_two(self, x_train, y_train, 
                                          minority_classes=[7, 8],
                                          concentration=0.8, 
                                          non_minority_fraction_for_client0=0.1,
                                          batch_size=128):
        """
        Splits data among num_clients such that for each minority class in minority_classes:
          - 80% of its samples are assigned to client 0.
          - The remaining 20% is split equally between client 1 and client 2.
        Then, non-minority data is distributed as follows:
          - A small fraction (non_minority_fraction_for_client0) is added to client 0.
          - The remaining non-minority data is evenly distributed among clients 1 to num_clients.
        
        :param x_train: NumPy array of training samples.
        :param y_train: NumPy array of training labels.
        :param minority_classes: List of minority class labels (e.g., [7, 8]).
        :param concentration: Fraction (e.g., 0.8) of samples for each minority class to assign to client 0.
        :param non_minority_fraction_for_client0: Fraction of non-minority data to assign to client 0.
        :param batch_size: Batch size for each DataLoader.
        :return: List of DataLoaders (one per client).
        """
        num_clients = self.num_clients
        if num_clients < 3:
            raise ValueError("num_clients must be at least 3.")
        
        # Initialize lists for each client.
        client_indices = [[] for _ in range(num_clients)]
        
        # Process each minority class.
        for cls in minority_classes:
            cls_idx = np.where(y_train == cls)[0]
            np.random.shuffle(cls_idx)
            n_total = len(cls_idx)
            n_concentrated = int(concentration * n_total)  # 80% for client 0
            client_indices[0].extend(cls_idx[:n_concentrated])
            
            # The remaining 20% is split equally between client 1 and client 2.
            leftover = cls_idx[n_concentrated:]
            if len(leftover) > 0:
                splits = np.array_split(leftover, 2)
                client_indices[1].extend(splits[0])
                client_indices[2].extend(splits[1])
        
        # Process non-minority data.
        non_minority_idx = np.where(~np.isin(y_train, minority_classes))[0]
        np.random.shuffle(non_minority_idx)
        
        # Split non-minority data into two parts:
        # - A small fraction for client 0.
        # - The rest to be evenly distributed among clients 1 to num_clients.
        n_for_client0 = int(non_minority_fraction_for_client0 * len(non_minority_idx))
        client0_non_minority = non_minority_idx[:n_for_client0]
        remaining_non_minority = non_minority_idx[n_for_client0:]
        
        # Add non-minority data to client 0.
        client_indices[0].extend(client0_non_minority)
        
        # Distribute the remaining non-minority data evenly among clients 1 to num_clients.
        num_other_clients = num_clients - 1  # clients 1,...,num_clients
        splits_remaining = np.array_split(remaining_non_minority, num_other_clients)
        for i in range(1, num_clients):
            client_indices[i].extend(splits_remaining[i-1])
        
        # Shuffle indices within each client.
        for i in range(num_clients):
            np.random.shuffle(client_indices[i])
        
        # Create DataLoader for each client.
        data_loaders = []
        for i in range(num_clients):
            indices = np.array(client_indices[i])
            x_client = x_train[indices]
            y_client = y_train[indices]
            
            # If x_train has shape (N, H, W), expand dims to (N, 1, H, W) for PyTorch CNNs.
            if x_client.ndim == 3:
                x_client = np.expand_dims(x_client, axis=1)
            # If x_train is (N, H, W, C) and channels are last, you might need to transpose to (N, C, H, W).
            # x_client = np.transpose(x_client, (0, 3, 1, 2))
            
            dataset = TensorDataset(torch.tensor(x_client, dtype=torch.float32),
                                    torch.tensor(y_client, dtype=torch.long))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            data_loaders.append(loader)
        
        return data_loaders


    def get_test_dataloader(self, x_test, y_test, batch_size=1000):
        """
        Prepares the test DataLoader from the test data.

        :param x_test: Test data
        :type x_test: np.ndarray
        :param y_test: Test labels
        :type y_test: np.ndarray
        :param batch_size: Batch size for the DataLoader
        :type batch_size: int
        :return: A DataLoader for the test data
        :rtype: DataLoader
        """
        x_test_pt = np.expand_dims(x_test, axis=1)
        y_test_pt = y_test.astype(int)
        
        # Create TensorDataset
        dataset_test = TensorDataset(torch.Tensor(x_test_pt), torch.Tensor(y_test_pt).long())
        
        # Create DataLoader without shuffling
        testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        
        return testloader
        