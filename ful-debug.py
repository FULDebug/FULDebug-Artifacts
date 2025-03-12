from diskcache import Cache
import os
from Utils.data_manager import DataManager
from Utils.models import FMNIST_Net
from Clients.training import Training
import copy
import torch
from torch import nn
from Utils.analytics import Analytics
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import signal
import time
from Ful_Algo.ibm_ful import IBMFUL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_paused = False
breakpoint_set = False
step_mode = False
cache = Cache("./cache")

class FULDebug:
    def __init__(self, cache, breakpoint, num_parties, num_fl_rounds):
        
        cache["breakpoint"] = breakpoint
        self.num_parties = num_parties
        self.num_fl_rounds = num_fl_rounds
        self.cache = cache
    
    def average_selected_models(self, selected_parties, party_models):
        with torch.no_grad():
            sum_vec = nn.utils.parameters_to_vector(party_models[selected_parties[0]].parameters())
            if len(selected_parties) > 1:
                for i in range(1,len(selected_parties)):
                    sum_vec += nn.utils.parameters_to_vector(party_models[selected_parties[i]].parameters())
                sum_vec /= len(selected_parties)

            model = copy.deepcopy(party_models[0])
            nn.utils.vector_to_parameters(sum_vec, model.parameters())
        return model.state_dict()
    
    def aggregate(self, client_models, current_model=None):
        selected_parties = [i for i in range(self.num_parties)]
        aggregated_model_state_dict = self.average_selected_models(selected_parties, client_models)
        return aggregated_model_state_dict 
    
    def partiesStart(self, trainloader_lst, testloader, client_to_be_erased=100, dataType="FMNIST"):

        num_fl_rounds = self.num_fl_rounds
        num_parties = self.num_parties
        initial_model = FMNIST_Net()
        model_dict = copy.deepcopy(initial_model.state_dict())
        for round_num in range(num_fl_rounds): 
            ##################### Local Training Round #############################
            current_model_state_dict = copy.deepcopy(model_dict)
            current_model = copy.deepcopy(initial_model)
            current_model.load_state_dict(current_model_state_dict)
            client_models = []
            party_losses = []
            for party_id in range(num_parties):

                if party_id == client_to_be_erased:
                    client_models.append(FMNIST_Net())
                else:
                    model = copy.deepcopy(current_model)
                    local_training = Training(num_updates_in_epoch=None, num_local_epochs=1)
                    model_update, party_loss = local_training.train(model=model, 
                                                trainloader=trainloader_lst[party_id], 
                                                criterion=None, opt=None, dataType=dataType)

                    client_models.append(copy.deepcopy(model_update))
                    party_losses.append(party_loss)
                    print(f"Party {party_id} Loss: {party_loss}")
            ######################################################################  
            current_model_state_dict = self.aggregate(client_models=client_models, current_model=current_model)
            model_dict = copy.deepcopy(current_model_state_dict)
            eval_model = FMNIST_Net()
            eval_model.load_state_dict(current_model_state_dict)
            clean_acc = local_training.evaluate(testloader, eval_model)
            # clean_accuracy[fusion_key][round_num] = clean_acc        
            self.cache[f"client_models"] = client_models
            self.cache[f"global_models"] = current_model_state_dict
            print(f'Global Clean Accuracy, round {round_num} = {clean_acc}')
            # print(self.cache.get(f"client_models_{round_num}"))

    # Function to compute class-wise accuracy
    def compute_classwise_metrics(self, model, test_loader):
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        model.eval()
        
        with torch.no_grad():
            for data, labels in test_loader:
                if data.dim() == 5:  # If the image has an extra dimension, squeeze it
                    data = data.squeeze(1)  # Remove the extra dimension
                    
                # Check if the input tensor has the correct shape for CIFAR-10
                if data.shape[1] == 32:  # Indicates the channel dimension is incorrectly set as 32
                    # Permute from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                    data = data.permute(0, 3, 1, 2)
                    
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                
                # Update class-wise correct/total counts
                for label, prediction in zip(labels, predicted):
                    class_total[label.item()] += 1
                    if label.item() == prediction.item():
                        class_correct[label.item()] += 1
        
        # Compute class-wise accuracies
        class_accuracies = {cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
                            for cls in class_total}
        
        return class_accuracies, class_total

    # Function to identify classes impacted by unlearning
    def identify_affected_classes(self, global_model_before, global_model_after, test_loader, threshold=0.05):
        """
        Identify the classes that are significantly impacted after unlearning client 0.
        
        :param global_model_before: The global model before unlearning.
        :param global_model_after: The global model after unlearning.
        :param test_loader: The test data loader.
        :param threshold: The threshold to consider a class significantly impacted.
        :return: A list of impacted classes.
        """
        # Compute class-wise accuracy before and after unlearning
        before_class_accuracies, _ = self.compute_classwise_metrics(global_model_before, test_loader)
        after_class_accuracies, _ = self.compute_classwise_metrics(global_model_after, test_loader)
    
        impacted_classes = []
    
        # Compare class accuracies before and after unlearning
        for cls in before_class_accuracies:
            accuracy_drop = before_class_accuracies[cls] - after_class_accuracies[cls]
            if accuracy_drop > threshold:
                impacted_classes.append(cls)
    
        return impacted_classes

    def calculate_class_weights(self, global_model_before, global_model_after, test_loader, impacted_classes):
        """
        Calculate class weights based on the accuracy difference before and after unlearning.
        
        :param global_model_before: Global model before unlearning.
        :param global_model_after: Global model after unlearning.
        :param test_loader: DataLoader for the test data.
        :param impacted_classes: List of impacted classes.
        :return: Dictionary with class indices as keys and weights as values.
        """
        # Compute class-wise accuracies before and after unlearning
        class_accuracies_before, _ = self.compute_classwise_metrics(global_model_before, test_loader)
        class_accuracies_after, _ = self.compute_classwise_metrics(global_model_after, test_loader)
        
        # Calculate the absolute difference in accuracy for each impacted class
        accuracy_diffs = {class_idx: abs(class_accuracies_before[class_idx] - class_accuracies_after[class_idx])
                          for class_idx in impacted_classes}
        
        # Normalize the differences to sum to 1 (to be used as weights)
        total_diff = sum(accuracy_diffs.values())
        class_weights = {class_idx: (diff / total_diff) for class_idx, diff in accuracy_diffs.items()} if total_diff > 0 else {class_idx: 1/len(impacted_classes) for class_idx in impacted_classes}
        
        # Print class weights for reference
        for class_idx, weight in class_weights.items():
            print(f"Class {class_idx} Weight: {weight:.4f}")
        
        return class_weights

    
    def select_clients_to_fix_bias(self, clients_models, impacted_classes, test_loader, global_model, global_model_before, global_model_after, num_clients=3, lambda_penalty=0.1):
        """
        Select clients that contribute the most to the affected classes with automated class weighting and regularization.
        
        :param clients_models: List of models for remaining clients.
        :param impacted_classes: List of classes impacted by unlearning client 0.
        :param test_loader: DataLoader for the test data.
        :param global_model: Global model to compute deviations for regularization.
        :param global_model_before: Global model before unlearning.
        :param global_model_after: Global model after unlearning.
        :param num_clients: Number of clients to select for fixing the bias.
        :param lambda_penalty: Regularization term to penalize clients with large deviations from global performance.
        :return: Tuple of (list of selected client indices, list of selected client models).
        """
        # Automatically assign class weights based on accuracy impact
        class_weights = self.calculate_class_weights(global_model_before, global_model_after, test_loader, impacted_classes)
    
        client_contributions = []
        
        # Compute global model's class-wise accuracy for regularization
        global_class_accuracies, _ = self.compute_classwise_metrics(global_model, test_loader)
        
        # Compute class-wise accuracy for each client
        for client_idx, client_model in enumerate(clients_models):
            class_accuracies, _ = self.compute_classwise_metrics(client_model, test_loader)
            
            # Calculate the weighted contribution of this client to the impacted classes
            contribution = sum((class_accuracies[class_idx] * class_weights[class_idx]) for class_idx in impacted_classes)
            
            # Compute the regularization term: deviation from the global model
            deviation_penalty = sum(abs(class_accuracies[class_idx] - global_class_accuracies[class_idx]) for class_idx in impacted_classes)
            
            # Final score: contribution minus regularization penalty
            final_contribution = contribution - lambda_penalty * deviation_penalty
            
            client_contributions.append((client_idx, client_model, final_contribution))
        
        # Sort clients by their final contribution score
        client_contributions.sort(key=lambda x: x[2], reverse=True)
        
        # Select top clients to fix the bias
        selected_clients = client_contributions[:num_clients]
        
        # Extract client indices and models for the selected clients
        selected_client_indices = [client_idx for client_idx, _, _ in selected_clients]
        selected_client_models = [client_model for _, client_model, _ in selected_clients]
        
        # Print selected clients and their contributions
        for client_idx, _, contribution in selected_clients:
            print(f"Selected Client {client_idx} with Contribution: {contribution:.4f}")
        
        # Return both selected client indices and models
        return selected_client_models, selected_client_indices


    def unlearnedModelAggregationWithSelectedClients(self, trainloader_lst, testDataloader, unlearned_model, num_rounds=10, client_to_be_erased=100, select_clients_method='random', select_num_clients=6):
    
        num_parties = self.num_parties
        initial_model = FMNIST_Net()
        current_model_state_dict = copy.deepcopy(unlearned_model.state_dict())  
        initial_model.load_state_dict(current_model_state_dict)  
        model_dict = copy.deepcopy(initial_model.state_dict())

        model_before = self.cache.get('initial_model')
        model_before.load_state_dict(self.cache.get('global_models'))
        model_before.eval()
        
        model_after = self.cache.get('initial_model')
        model_after.load_state_dict(self.cache.get("unlearned_model").state_dict())
        model_after.eval()
        
        client_model = self.cache.get('initial_model')
        client_model.load_state_dict(self.cache.get("client_models")[0].state_dict())
        client_model.eval()
        
        global_model_accuracies = []
        start_round = cache.get('round_num', 0)
        for round_num in range(start_round, num_rounds): 
            ##################### Local Training Round #############################
            current_model_state_dict = copy.deepcopy(model_dict)
            current_model = copy.deepcopy(initial_model)
            current_model.load_state_dict(current_model_state_dict)
            client_models = []
            party_losses = []
            # Load the last round if paused, or start from round 0
            for party_id in range(num_parties):
                if party_id == client_to_be_erased:
                    client_models.append(FMNIST_Net())  # Placeholder for unlearned client
                else:
                    model = copy.deepcopy(current_model)
                    local_training = Training(num_updates_in_epoch=None, num_local_epochs=1)
                    model_update, party_loss = local_training.train(model=model, 
                                                    trainloader=trainloader_lst[party_id], 
                                                    criterion=None, opt=None, dataType="Fashion")
    
                    client_models.append(copy.deepcopy(model_update))
                    party_losses.append(party_loss)
                    print(f"Party {party_id} Loss: {party_loss}")
    
            ###################### Client Selection ###############################
            impacted_classes = self.identify_affected_classes(model_before, model_after, testDataloader, threshold=0.05)
            selected_client_models, selected_client_indices = self.select_clients_to_fix_bias(client_models, impacted_classes, testDataloader, model_before, model_before, model_after, num_clients=5)
            print(selected_client_models)
    
            #######################################################################
            # Aggregate only the selected client models
            current_model_state_dict = self.unlearnAggregate(client_models=selected_client_models, client_to_be_erased=client_to_be_erased)
            model_dict = copy.deepcopy(current_model_state_dict)
            eval_model = FMNIST_Net()
            eval_model.load_state_dict(current_model_state_dict)
    
            clean_acc = local_training.evaluate(testDataloader, eval_model)
            print(f'Global Clean Accuracy, round {round_num} = {clean_acc}')
            global_model_accuracies.append(clean_acc)
            self.cache[f"unlearning_client_models"] = client_models
            self.cache[f"unlearning_global_models"] = current_model_state_dict
            cache.set('round_num', round_num)
            # Check if the loop should pause after each round
            if is_paused:
                print(f"Loop paused at round {round_num}.")
                cache.set('round_num', int(start_round + 1))
                break
        return global_model_accuracies
    
    def unlearnAggregate(self, client_models, client_to_be_erased):
        selected_parties = [i for i in range(len(client_models))]
        aggregated_model_state_dict = self.average_selected_models(selected_parties, client_models)
        return aggregated_model_state_dict
    
    def unlearnedModelAggregation(self, trainloader_lst, testloader, unlearned_model, num_rounds=10, client_to_be_erased=100):

        num_parties = self.num_parties
        initial_model = FMNIST_Net()
        current_model_state_dict = copy.deepcopy(unlearned_model.state_dict())  
        initial_model.load_state_dict(current_model_state_dict)  
        model_dict = copy.deepcopy(initial_model.state_dict())
        # Load the last round if paused, or start from round 0
        for round_num in range(num_rounds): 
            ##################### Local Training Round #############################
            current_model_state_dict = copy.deepcopy(model_dict)
            current_model = copy.deepcopy(initial_model)
            current_model.load_state_dict(current_model_state_dict)
            client_models = []
            party_losses = []
            for party_id in range(num_parties):

                if party_id == client_to_be_erased:
                    client_models.append(FMNIST_Net())
                else:
                    model = copy.deepcopy(current_model)
                    local_training = Training(num_updates_in_epoch=None, num_local_epochs=1)
                    model_update, party_loss = local_training.train(model=model, 
                                                trainloader=trainloader_lst[party_id], 
                                                criterion=None, opt=None, dataType="FMNIST")

                    client_models.append(copy.deepcopy(model_update))
                    party_losses.append(party_loss)
                    print(f"Party {party_id} Loss: {party_loss}")
            ######################################################################  
            current_model_state_dict = self.unlearnAggregate(client_models=client_models, client_to_be_erased=client_to_be_erased)
            model_dict = copy.deepcopy(current_model_state_dict)
            eval_model = FMNIST_Net()
            eval_model.load_state_dict(current_model_state_dict)
            clean_acc = local_training.evaluate(testloader, eval_model)
            # clean_accuracy[fusion_key][round_num] = clean_acc      
            self.cache[f"unlearning_client_models"] = client_models
            self.cache[f"unlearning_global_models"] = current_model_state_dict
            print(f'Global Clean Accuracy, round {round_num} = {clean_acc}')

    def compute_weight_contribution(self, global_model, client_updates, selected_client_idx):
        """
        Compute the influence of each client's weight contribution to the global model.

        :param global_model: The baseline global model (PyTorch model)
        :param client_updates: List of model updates from each client (list of state_dicts)
        :param selected_client_idx: Index of the client whose contribution you want to analyze
        :return: Difference between the global model's weights with and without the selected client's contribution
        """
        # Compute the average weight update with all clients
        num_clients = len(client_updates)
        avg_update = {key: torch.zeros_like(val) for key, val in client_updates[0].state_dict().items()}

        for update in client_updates:
            for key in update.state_dict():
                avg_update[key] += update.state_dict()[key] / num_clients

        # Compute the average weight update without the selected client
        avg_update_without_client = {key: torch.zeros_like(val) for key, val in client_updates[0].state_dict().items()}

        for i, update in enumerate(client_updates):
            if i == selected_client_idx:
                continue  # Skip the selected client
            for key in update.state_dict():
                avg_update_without_client[key] += update.state_dict()[key] / (num_clients - 1)

        # Calculate the difference in the global model's weights
        weight_difference = {key: avg_update[key] - avg_update_without_client[key] for key in avg_update}

        return weight_difference

    def compute_weight_norm_difference(self, weight_difference):
        """
        Computes the norm of the weight differences to quantify the impact.

        :param weight_difference: Dictionary containing weight differences for each layer
        :return: Dictionary with norms for each layer
        """
        norm_diff = {}
        for layer, diff in weight_difference.items():
            norm_diff[layer] = torch.norm(diff).item()
        return norm_diff
    
    def analyze_class_bias(self, global_model, weight_difference, num_classes=10):
        """
        Analyzes the class-specific impact of removing a client's weight contribution.

        :param global_model: The baseline global model (PyTorch model)
        :param weight_difference: Difference in weights with and without the selected client's contribution
        :param num_classes: Number of classes in the dataset (e.g., 10 for MNIST)
        :return: Impact on each class based on output layer weight differences
        """
        output_layer_key = None

        # Identify the output layer by checking for the appropriate layer name
        for key in weight_difference.keys():
            if 'weight' in key and weight_difference[key].shape[0] == num_classes:
                output_layer_key = key
                break

        if output_layer_key is None:
            raise ValueError("Could not identify the output layer in the model.")

        # Analyze the impact on each class
        class_impact = torch.norm(weight_difference[output_layer_key], dim=1).tolist()
        return class_impact

    def summarize_and_print_results(slef, norm_diff, class_impact):
        """
        Summarizes and prints the results of the weight differences and class impacts.

        :param norm_diff: Dictionary containing the norms of weight differences for each layer
        :param class_impact: List containing the impact on each class
        """
        print("=== Summary of Weight Differences by Layer ===")
        print(f"{'Layer':<20} {'Norm Difference':>20}")
        print("-" * 40)
        
        for layer, norm in norm_diff.items():
            print(f"{layer:<20} {norm:>20.6f}")
        
        print("\n=== Impact on Each Class ===")
        print(f"{'Class':<10} {'Impact':>10}")
        print("-" * 25)
        
        for class_idx, impact in enumerate(class_impact):
            print(f"Class {class_idx:<5} {impact:>10.6f}")
        
        print("\n=== Analysis ===")
        
        # Find the class with the maximum and minimum impact
        max_impact_class = max(range(len(class_impact)), key=lambda i: class_impact[i])
        min_impact_class = min(range(len(class_impact)), key=lambda i: class_impact[i])
        
        print(f"The highest impact is on Class {max_impact_class} with an impact value of {class_impact[max_impact_class]:.6f}.")
        print(f"The lowest impact is on Class {min_impact_class} with an impact value of {class_impact[min_impact_class]:.6f}.")

        # Determine which layers are most and least affected
        most_affected_layer = max(norm_diff, key=norm_diff.get)
        least_affected_layer = min(norm_diff, key=norm_diff.get)
        
        print(f"The most affected layer is '{most_affected_layer}' with a norm difference of {norm_diff[most_affected_layer]:.6f}.")
        print(f"The least affected layer is '{least_affected_layer}' with a norm difference of {norm_diff[least_affected_layer]:.6f}.")
        print("\nThis analysis suggests that removing the selected client's contribution mainly affects the above class and layer.")



def signal_handler(signal, frame):
    print("Breakpoint received. Pausing execution...")
    global is_paused
    is_paused = True

signal.signal(signal.SIGINT, signal_handler) 

def debugger_interface():
    while True:
        print("Debugger Menu:")
        print("1. Step-In")
        print("2. Step-Out")
        print("3. Resume")
        print("4. Set Breakpoint")
        choice = input("Enter choice: ")
        global is_paused, step_mode
        if choice == "1":
            step_mode = True
            is_paused = True
            print("Stepping into the loop...")
            return
        elif choice == "2":
            step_mode = False
            is_paused = False
            print("Stepping out...")
            return
        elif choice == "3":
            is_paused = False
            print("Resuming execution...")
            return
        elif choice == "4":
            set_breakpoint()
        else:
            print("Invalid choice. Try again.")

def set_breakpoint():
    global breakpoint_set
    breakpoint_set = True
    print("Breakpoint set for the next loop.")

def display_menu():
    options = [
        "Option 1: Step-In",
        "Option 2: Step-Out",
        "Option 3: Resume"
    ]
    
    # Display the options with numbers
    print("Please select an option:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    # Ask the user to enter the number
    try:
        choice = int(input("Enter the number to view Analytics: "))
        if 1 <= choice <= len(options):
            return choice
        else:
            print("Invalid input, please enter a number between 1 and 4.")
            return display_menu()  # Recursively call the function to ask again
    except ValueError:
        print("Invalid input, please enter a valid number.")
        return display_menu()

def display_analytical_menu_before_unlearn():
    options = [
        "Option 1: Client vs Global Per Class Accuracy",
        "Option 2: Display Client Contributions to Class Compared to Others",
        "Option 3: Plot Client Contributions to Class Compared to Others",
        "Option 4: Step-Out"
    ]
    
    # Display the options with numbers
    print("Please select an option:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    # Ask the user to enter the number
    try:
        choice = int(input("Enter the number to view Analytics: "))
        if 1 <= choice <= len(options):
            return choice
        else:
            print("Invalid input, please enter a number between 1 and 4.")
            return display_analytical_menu_before_unlearn()  # Recursively call the function to ask again
    except ValueError:
        print("Invalid input, please enter a valid number.")
        return display_analytical_menu_before_unlearn()

def display_analytical_menu_after_unlearn():
    options = [
        "Option 1: Client vs Global Per Class Accuracy",
        "Option 2: Display Client Contributions to Class Compared to Others",
        "Option 3: Plot Client Contributions to Class Compared to Others",
        "Option 4: Step-Out"
    ]
    
    # Display the options with numbers
    print("Please select an option:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    # Ask the user to enter the number
    try:
        choice = int(input("Enter the number to view Analytics: "))
        if 1 <= choice <= len(options):
            return choice
        else:
            print("Invalid input, please enter a number between 1 and 4.")
            return display_analytical_menu_after_unlearn()  # Recursively call the function to ask again
    except ValueError:
        print("Invalid input, please enter a valid number.")
        return display_analytical_menu_after_unlearn()
    
# Main function
def main():
    # Initialize the cache
    cache.clear()
    initial_model = FMNIST_Net()
    cache["initial_model"] = initial_model
    breakpoint =  {"round": 5, "status": False}
    # Initialize the MNIST loader
    parties = int(input("Please enter number of clients: "))
    loader = DataManager(download_dir="./data", normalize=True, num_clients=parties)
    # Load the MNIST data
    (x_train, y_train), (x_test, y_test) = loader.load_fashion_mnist()
    dataLoader = loader.split_data_uneven(x_train, y_train)
    testDataloader = loader.get_test_dataloader(x_test, y_test)
    rounds = int(input("Please enter number of FL Rounds: "))
    # Example usage
    sim = FULDebug(cache, breakpoint, parties, rounds)
    sim.partiesStart(dataLoader, testDataloader)
    
    choice = display_menu()
    
    if(choice == 1):
        # Step-In
        while True:
            analytical_choice = display_analytical_menu_before_unlearn()
            if analytical_choice == 4:
                break
            else:
                if analytical_choice == 1:
                    # Display Client vs Global Per Class Accuracy
                    analytics = Analytics(cache)
                    client_idx = int(input("Please enter Client idx: "))
                    analytics.client_vs_global_per_class_accuracy(client_idx, testDataloader)
                elif analytical_choice == 2:
                    # Display Client Contributions to Class Compared to
                    analytics = Analytics(cache)
                    client_idx = int(input("Please enter Client idx: "))
                    analytics.display_client_contributions_to_calss_compared_to_others(client_idx, testDataloader, num_clients=parties)
                elif analytical_choice == 3:
                    # Plot Client Contributions to Class Compared to
                    analytics = Analytics(cache)
                    client_idx = int(input("Please enter Client idx: "))
                    analytics.plot_client_contributions_to_calss_compared_to_others(client_idx, testDataloader, num_clients=parties)

    signal.signal(signal.SIGINT, signal_handler)
    
    unlearning_instance = IBMFUL(
        client_models=cache.get("client_models"),
        global_model=cache.get("global_models"),
        num_parties=10,  # Example value
        party_to_be_erased=0,  # Example value
        trainloader_lst=dataLoader,
        testloader=testDataloader,
        testloader_poison=dataLoader,
        initial_model=FMNIST_Net(),
        lr=0.01,
        num_local_epochs_unlearn=5,
        distance_threshold=0.6,
        clip_grad=5
    )

    unlearned_model = unlearning_instance.execute_unlearning()
    cache["unlearned_model"] = unlearned_model

    sim.unlearnedModelAggregationWithSelectedClients(dataLoader, testDataloader, unlearned_model, num_rounds=5, client_to_be_erased=0)

    if is_paused:
        debugger_interface()

    time.sleep(4)
    # Step-In or Step-ut Logic
    while step_mode:
        analytical_choice = input("Select Analytics: 1. Client vs Global Per Class Accuracy, 2. Client Contributions, 3. Plot Contributions: ")
        client_idx = int(input("Enter client index: "))
        if analytical_choice == "1":
            analytics.client_vs_global_per_class_accuracy(client_idx, testDataloader)
        elif analytical_choice == "2":
            analytics.display_client_contributions_to_class_compared_to_others(client_idx, testDataloader, num_clients=parties)
        elif analytical_choice == "3":
            analytics.plot_client_contributions_to_class_compared_to_others(client_idx, testDataloader, num_clients=parties)
        step_choice = input("Step-In to continue (y/n): ")
        if step_choice.lower() != "y":
            step_mode = False


# Ensure the main function runs only when executed directly
if __name__ == "__main__":
    main()