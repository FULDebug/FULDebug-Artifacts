import torch
import torch.nn as nn
import copy
import numpy as np
from Utils.models import CIFAR_Net

class IBMFUL:
    def __init__(self, client_models, global_model, num_parties, party_to_be_erased, trainloader_lst, testloader, testloader_poison, initial_model, lr=0.01, num_local_epochs_unlearn=5, distance_threshold=2.5, clip_grad=5):
        self.client_models = client_models
        self.global_model = global_model
        self.num_parties = num_parties
        self.party_to_be_erased = party_to_be_erased
        self.trainloader_lst = trainloader_lst
        self.testloader = testloader
        self.testloader_poison = testloader_poison
        self.initial_model = initial_model
        self.lr = lr
        self.num_local_epochs_unlearn = num_local_epochs_unlearn
        self.distance_threshold = distance_threshold
        self.clip_grad = clip_grad

    def compute_reference_model(self, fedavg_model, unlearned_client_model):
        initial_model = self.initial_model
        fedavg_model_state_dict = copy.deepcopy(fedavg_model)
        fedavg_model = copy.deepcopy(initial_model)
        fedavg_model.load_state_dict(fedavg_model_state_dict)

        #compute reference model
        #w_ref = N/(N-1)w^T - 1/(N-1)w^{T-1}_i = \sum{i \ne j}w_j^{T-1}
        model_ref_vec = self.num_parties / (self.num_parties - 1) * nn.utils.parameters_to_vector(fedavg_model.parameters()) \
                        - 1 / (self.num_parties - 1) * nn.utils.parameters_to_vector(unlearned_client_model.parameters())

        model_ref = copy.deepcopy(self.initial_model)
        nn.utils.vector_to_parameters(model_ref_vec, model_ref.parameters())
        return model_ref

    def get_distance(self, model1, model2):
        with torch.no_grad():
            model1_flattened = nn.utils.parameters_to_vector(model1.parameters())
            model2_flattened = nn.utils.parameters_to_vector(model2.parameters())
            distance = torch.square(torch.norm(model1_flattened - model2_flattened))
        return distance
    
    def get_distances_from_current_model(self, current_model, party_models):
        num_updates = len(party_models)
        distances = np.zeros(num_updates)
        for i in range(num_updates):
            distances[i] = self.get_distance(current_model, party_models[i])
        return distances
    
    def calculate_threshold(self, model_ref):
        dist_ref_random_lst = [self.get_distance(model_ref, self.initial_model) for _ in range(10)]
        mean_distance = np.mean(dist_ref_random_lst)
        print(f'Mean distance of Reference Model to random: {mean_distance}')
        threshold = mean_distance / 3
        print(f'Radius for model_ref: {threshold}')
        return threshold

    def perform_unlearning(self, model_ref, party0_model, threshold):
        model = copy.deepcopy(model_ref)
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        model.train()

        for epoch in range(self.num_local_epochs_unlearn):
            print('------------', epoch)
            for batch_id, (x_batch, y_batch) in enumerate(self.trainloader_lst[self.party_to_be_erased]):
                if len(x_batch.shape) == 5:  # If shape is [batch_size, 1, 1, 28, 28]
                    x_batch = x_batch.squeeze(1)
                # Check if the input tensor has the correct shape for CIFAR-10
                if x_batch.shape[1] == 32:  # Indicates the channel dimension is incorrectly set as 32
                    # Permute from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                    x_batch = x_batch.permute(0, 3, 1, 2)

                opt.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss_joint = -loss  # negate the loss for gradient ascent
                loss_joint.backward()

                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)

                opt.step()

                with torch.no_grad():
                    distance = self.get_distance(model, model_ref)
                    if distance > threshold:
                        dist_vec = nn.utils.parameters_to_vector(model.parameters()) - nn.utils.parameters_to_vector(model_ref.parameters())
                        dist_vec = dist_vec / torch.norm(dist_vec) * np.sqrt(threshold)
                        proj_vec = nn.utils.parameters_to_vector(model_ref.parameters()) + dist_vec
                        nn.utils.vector_to_parameters(proj_vec, model.parameters())
                        distance = self.get_distance(model, model_ref)

                distance_ref_party_0 = self.get_distance(model, party0_model)
                print('Distance from the unlearned model to party to be erased:', distance_ref_party_0.item())

                if distance_ref_party_0 > self.distance_threshold:
                    return model

        return model
    
    def evaluate(self, testloader, model):

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
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

    def execute_unlearning(self):
        # Step 1: Initialize initial models
        fedavg_model = copy.deepcopy(self.global_model)
        party_models = copy.deepcopy(self.client_models)  # Assuming the key is 'Retrain'
        party0_model = copy.deepcopy(party_models[self.party_to_be_erased])

        # Step 2: Compute reference model
        model_ref = self.compute_reference_model(fedavg_model, party0_model)

        # Step 3: Calculate threshold
        threshold = self.calculate_threshold(model_ref)

        # Step 4: Perform unlearning
        unlearned_model = self.perform_unlearning(model_ref, party0_model, threshold)

        # Step 5: Evaluate the unlearned model
        eval_model = self.initial_model
        eval_model.load_state_dict(unlearned_model.state_dict())
        unlearn_clean_acc = self.evaluate(self.testloader, eval_model)
        print(f'Clean Accuracy for UN-Local Model = {unlearn_clean_acc}')
        # unlearn_clean_acc = Utils.evaluate(self.testloader, eval_model)
        # print(f'Clean Accuracy for UN-Local Model = {unlearn_clean_acc}')
        # pois_unlearn_acc = Utils.evaluate(self.testloader_poison, eval_model)
        # print(f'Backdoor Accuracy for UN-Local Model = {pois_unlearn_acc}')

        return unlearned_model