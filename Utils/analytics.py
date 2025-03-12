import torch
from torch import nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import shap
import copy
from matplotlib.patches import Patch

class Analytics:

    """
    Base class for Analytics
    """

    def __init__(self, 
                 cache,
                 num_updates_in_epoch=None,
                 num_local_epochs=1):
        
        self.name = "analytics"
        self.cache = cache
        self.num_local_epochs = num_local_epochs

    def print_cache(self):
        print(self.cache._sql('SELECT key FROM Cache').fetchall())

    def evaluate_model(self, model, dataloader):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                # Ensure that the images are of shape [batch_size, channels, height, width]
                if len(images.shape) == 5:  # If shape is [batch_size, 1, 1, 28, 28]
                    images = images.squeeze(1)  # Squeeze the second dimension [batch_size, 1, 28, 28]
                
                if images.shape[1] == 32:  # Indicates the channel dimension is incorrectly set as 32
                    # Permute from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                    images = images.permute(0, 3, 1, 2)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = correct / total
        return accuracy, all_preds, all_labels
    
    def per_class_accuracy(self, labels, preds, num_classes=10):
        confusion = confusion_matrix(labels, preds, labels=range(num_classes))
        per_class_acc = confusion.diagonal() / confusion.sum(axis=1)
        return per_class_acc
    

    def client_vs_global_per_class_accuracy(self, client_idx, testDataloader):
        
        global_model_before = self.cache.get('initial_model')
        global_model_before.load_state_dict(self.cache.get('global_models'))
        global_model_before.eval()

        client_model = self.cache.get('initial_model')
        client_model.load_state_dict(self.cache.get("client_models")[client_idx].state_dict())
        client_model.eval()

        client_accuracy, client_preds, client_labels = self.evaluate_model(client_model, testDataloader)
        global_accuracy_before, global_preds_before, global_labels_before = self.evaluate_model(global_model_before, testDataloader)

        client_per_class_acc = self.per_class_accuracy(client_labels, client_preds)
        global_per_class_acc_before = self.per_class_accuracy(global_labels_before, global_preds_before)

        print("### Client {client_idx} vs Global Model Per Class Accuracy ###")
        
        print(f"Client Model Accuracy = {client_accuracy*100:.2f}%, Global Model Before Accuracy = {global_accuracy_before*100:.2f}%")
        # Print per-class accuracies
        for i, (c_acc, g_acc) in enumerate(zip(client_per_class_acc, global_per_class_acc_before)):
            print(f"Class {i}: Client Model Accuracy = {c_acc*100:.2f}%, Global Model Before Accuracy = {g_acc*100:.2f}%")
        
        print("######")
    
    def unlearned_model_vs_global_per_class_accuracy(self, testDataloader):
        
        global_model_before = self.cache.get('initial_model')
        global_model_before.load_state_dict(self.cache.get('global_models'))
        global_model_before.eval()

        unlearned_model = self.cache.get('initial_model')
        unlearned_model.load_state_dict(self.cache.get("unlearned_model").state_dict())
        unlearned_model.eval()

        unlearned_model_accuracy, unlearned_model_preds, unlearned_model_labels = self.evaluate_model(unlearned_model, testDataloader)
        global_accuracy_before, global_preds_before, global_labels_before = self.evaluate_model(global_model_before, testDataloader)

        unlearned_model_per_class_acc = self.per_class_accuracy(unlearned_model_labels, unlearned_model_preds)
        global_per_class_acc_before = self.per_class_accuracy(global_labels_before, global_preds_before)

        print("### Unlearned vs Global Model Per Class Accuracy ###")
        
        print(f"Unlearned Model Accuracy = {unlearned_model_accuracy*100:.2f}%, Global Model Before Accuracy = {global_accuracy_before*100:.2f}%")
        # Print per-class accuracies
        for i, (g_acc_before, g_acc_after) in enumerate(zip(global_per_class_acc_before, unlearned_model_per_class_acc)):
            delta = g_acc_after - g_acc_before
            print(f"Class {i}: Global Model Before = {g_acc_before*100:.2f}%, After = {g_acc_after*100:.2f}%, Δ = {delta*100:.2f}%")
        
        print("######")
    
    def global_before_vs_global_after_per_class_accuracy(self, testDataloader):
        
        global_model_before = self.cache.get('initial_model')
        global_model_before.load_state_dict(self.cache.get('global_models'))
        global_model_before.eval()

        global_model_after = self.cache.get('initial_model')
        global_model_after.load_state_dict(self.cache.get("unlearning_global_models"))
        global_model_after.eval()

        global_accuracy_before, global_preds_before, global_labels_before = self.evaluate_model(global_model_before, testDataloader)
        global_accuracy_after, global_preds_after, global_labels_after = self.evaluate_model(global_model_after, testDataloader)

        global_per_class_acc_before = self.per_class_accuracy(global_labels_before, global_preds_before)
        global_per_class_acc_after = self.per_class_accuracy(global_labels_after, global_preds_after)

        print("### Unlearned vs Global Model Per Class Accuracy ###")
        
        print(f"Global Model After Accuracy = {global_accuracy_after*100:.2f}%, Global Model Before Accuracy = {global_accuracy_before*100:.2f}%")
        # Print per-class accuracies
        for i, (g_acc_before, g_acc_after) in enumerate(zip(global_per_class_acc_before, global_per_class_acc_after)):
            delta = g_acc_after - g_acc_before
            print(f"Class {i}: Global Model Before = {g_acc_before*100:.2f}%, After = {g_acc_after*100:.2f}%, Δ = {delta*100:.2f}%")
        
        print("######")
    
    
    def apply_gradcam(self, model, images, target_layers):

        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=images)
        return grayscale_cam
    
    def visualize_comparison(self, image, cams, label, visualization_class=0, indices=None):
        
        if label.item() != visualization_class:
            return
        
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())
        image = np.repeat(image, 3, axis=2)  # Convert grayscale to RGB for visualization

        titles = ['Client Model', 'Global Model Before', 'Global Model After']
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(image.squeeze(), cmap='gray')
        axs[0].set_title(f'Original Image (Label: {label.item()})')

        for idx, cam in enumerate(cams):
            cam_image = show_cam_on_image(image, cam, use_rgb=True)
            axs[idx+1].imshow(cam_image)
            axs[idx+1].set_title(titles[idx])

        plt.show()

    
    def visualize_feature_comparison(self, client_idx, testDataloader, target_layers=None, num_images_to_visualize=5, visualization_class=0):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        global_model_before = self.cache.get('initial_model')
        global_model_before.load_state_dict(self.cache.get('global_models'))
        global_model_before.eval()

        global_model_after = self.cache.get('initial_model')
        global_model_after.load_state_dict(self.cache.get("unlearning_global_models"))
        global_model_after.eval()

        client_model = self.cache.get('initial_model')
        client_model.load_state_dict(self.cache.get("client_models")[client_idx].state_dict())
        client_model.eval()

        # Get a batch of test images
        dataiter = iter(testDataloader)
        images, labels = next(dataiter)
        images, labels = images.to(device), labels.to(device)
                
        if len(images.shape) == 5:  # If shape is [batch_size, 1, 1, 28, 28]
                images = images.squeeze(2)  # Squeeze the second dimension [batch_size, 1, 28, 28]
        
        if images.shape[1] == 32:  # Indicates the channel dimension is incorrectly set as 32
                # Permute from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                images = images.permute(0, 3, 1, 2)
                
        target_layers = [client_model.conv2]
        # Apply Grad-CAM to the client model
        grayscale_cams_client = self.apply_gradcam(client_model, images, target_layers)

        target_layers = [global_model_before.conv2]
        # # Apply Grad-CAM to the global model before unlearning
        grayscale_cams_before = self.apply_gradcam(global_model_before, images, target_layers)

        target_layers = [global_model_after.conv2]
        # # Apply Grad-CAM to the global model after unlearning
        grayscale_cams_after = self.apply_gradcam(global_model_after, images, target_layers)

        images_visualized = 0

        for i in range(len(images)):
            if labels[i].item() == visualization_class:
                cams = [
                    grayscale_cams_client[i],
                    grayscale_cams_before[i],
                    grayscale_cams_after[i]
                ]
                self.visualize_comparison(images[i], cams, labels[i], visualization_class=visualization_class, indices=[i])
                images_visualized += 1
            
            # Stop after visualizing the desired number of images
            if images_visualized >= num_images_to_visualize:
                break
    
    # Function to compute the feature change score between two models
    def compute_feature_change_score(self, cam1, cam2):
        # Compute the absolute difference between two Grad-CAM heatmaps
        feature_change = np.abs(cam1 - cam2)
        # Sum the differences to get a single score for each image
        feature_change_score = np.sum(feature_change)
        return feature_change_score
    
    def visualize_feature_change_class_wise_shared_unlearned_model(self, client_idx, testDataloader, num_classes=10):
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
            global_model_before = self.cache.get('initial_model')
            global_model_before.load_state_dict(self.cache.get('global_models'))
            global_model_before.eval()
    
            global_model_after = self.cache.get('initial_model')
            global_model_after.load_state_dict(self.cache.get("unlearned_model").state_dict())
            global_model_after.eval()
    
            # Get a batch of test images
            images, labels = next(iter(testDataloader))
            images, labels = images.to(device), labels.to(device)
            if len(images.shape) == 5:  # If shape is [batch_size, 1, 1, 28, 28]
                images = images.squeeze(1)
            if images.shape[1] == 32:  # Indicates the channel dimension is incorrectly set as 32
                # Permute from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                images = images.permute(0, 3, 1, 2)
                
            target_layers_before = [global_model_before.conv2]
            target_layers_after = [global_model_after.conv2]
            # Loop over all classes (0-9 for Fashion MNIST)
            feature_change_scores_global_vs_unlearned = []
            for i in range(num_classes):
                
                target_class_mask = labels == i
                if target_class_mask.sum() == 0:
                    continue

                # Get a batch of images for the target class
                target_images = images[target_class_mask]
                
                # Apply Grad-CAM to the global model before unlearning and the global model after unlearning
                grayscale_cams_before = self.apply_gradcam(global_model_before, target_images, target_layers_before)
                grayscale_cams_after = self.apply_gradcam(global_model_after, target_images, target_layers_after)
                
                # Compute the feature change score for the first image of this class
                score = self.compute_feature_change_score(grayscale_cams_before[0], grayscale_cams_after[0])
                
                # Append the score for this class
                feature_change_scores_global_vs_unlearned.append((i, score))

            # Sort feature change scores by class index
            feature_change_scores_global_vs_unlearned.sort(key=lambda x: x[0])

            # Unzip the feature change scores into separate lists for class indices and scores
            classes, scores = zip(*feature_change_scores_global_vs_unlearned)

            # Plot the feature change scores for all classes
            plt.figure(figsize=(10, 6))
            plt.bar(classes, scores, color='orange')
            plt.xlabel('Class')
            plt.ylabel('Feature Change Score')
            plt.title('Feature Change Score Between Global Model Before and Shared unlearned model')
            plt.xticks(classes)
            plt.show()

    def visualize_feature_change_class_wise(self, client_idx, testDataloader, num_classes=10):
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
            global_model_before = self.cache.get('initial_model')
            global_model_before.load_state_dict(self.cache.get('global_models'))
            global_model_before.eval()
    
            global_model_after = self.cache.get('initial_model')
            global_model_after.load_state_dict(self.cache.get("unlearning_global_models"))
            global_model_after.eval()
    
            # Get a batch of test images
            images, labels = next(iter(testDataloader))
            images, labels = images.to(device), labels.to(device)
            if len(images.shape) == 5:  # If shape is [batch_size, 1, 1, 28, 28]
                images = images.squeeze(1)
            if images.shape[1] == 32:  # Indicates the channel dimension is incorrectly set as 32
                # Permute from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                images = images.permute(0, 3, 1, 2)
                
            target_layers_before = [global_model_before.conv2]
            target_layers_after = [global_model_after.conv2]
            # Loop over all classes (0-9 for Fashion MNIST)
            feature_change_scores_global_vs_unlearned = []
            for i in range(num_classes):
                
                target_class_mask = labels == i
                if target_class_mask.sum() == 0:
                    continue

                # Get a batch of images for the target class
                target_images = images[target_class_mask]
                
                # Apply Grad-CAM to the global model before unlearning and the global model after unlearning
                grayscale_cams_before = self.apply_gradcam(global_model_before, target_images, target_layers_before)
                grayscale_cams_after = self.apply_gradcam(global_model_after, target_images, target_layers_after)
                
                # Compute the feature change score for the first image of this class
                score = self.compute_feature_change_score(grayscale_cams_before[0], grayscale_cams_after[0])
                
                # Append the score for this class
                feature_change_scores_global_vs_unlearned.append((i, score))

            # Sort feature change scores by class index
            feature_change_scores_global_vs_unlearned.sort(key=lambda x: x[0])

            # Unzip the feature change scores into separate lists for class indices and scores
            classes, scores = zip(*feature_change_scores_global_vs_unlearned)

            # Plot the feature change scores for all classes
            plt.figure(figsize=(10, 6))
            plt.bar(classes, scores, color='orange')
            plt.xlabel('Class')
            plt.ylabel('Feature Change Score')
            plt.title('Feature Change Score Between Global Model Before and After Unlearning')
            plt.xticks(classes)
            plt.show()
    
    # Function to calculate class-wise accuracy and confidence
    def calculate_classwise_metrics(self, model, dataloader, num_classes=10):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        class_correct = torch.zeros(num_classes)
        class_total = torch.zeros(num_classes)
        class_confidences = torch.zeros(num_classes)
        class_counts = torch.zeros(num_classes)

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                if len(images.shape) == 5:  # If shape is [batch_size, 1, 1, 28, 28]
                        images = images.squeeze(1)
                
                if images.shape[1] == 32:  # Indicates the channel dimension is incorrectly set as 32
                    # Permute from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                    images = images.permute(0, 3, 1, 2)

                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, dim=1)

                # Calculate per-class accuracy and confidence
                for i in range(num_classes):
                    mask = (labels == i)
                    class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
                    class_total[i] += mask.sum().item()
                    class_confidences[i] += probabilities[mask, i].sum().item()  # Confidence for correct class
                    class_counts[i] += mask.sum().item()  # Number of samples per class

        # Compute accuracy and average confidence for each class
        class_accuracies = class_correct / class_total
        avg_class_confidences = class_confidences / class_counts
        return class_accuracies, avg_class_confidences

    # Function to generate an ASCII bar for a given value
    def generate_ascii_bar(self,value, max_value, bar_length=30):
        filled_length = int(round(bar_length * value / max_value))
        return '|' * filled_length + '-' * (bar_length - filled_length)
   
    def display_classwise_metrics(self, client_idx, testDataloader, num_classes=10):

        global_model_before = self.cache.get('initial_model')
        global_model_before.load_state_dict(self.cache.get('global_models'))
        global_model_before.eval()

        global_model_after = self.cache.get('initial_model')
        global_model_after.load_state_dict(self.cache.get("unlearning_global_models"))
        global_model_after.eval()

        client_model = self.cache.get('initial_model')
        client_model.load_state_dict(self.cache.get("client_models")[client_idx].state_dict())
        client_model.eval()

        # Evaluate class-wise metrics for the client model
        class_accuracies_client, class_confidences_client = self.calculate_classwise_metrics(client_model, testDataloader)

        # Evaluate class-wise metrics for the global model before unlearning
        class_accuracies_before, class_confidences_before = self.calculate_classwise_metrics(global_model_before, testDataloader)

        # Evaluate class-wise metrics for the global model after unlearning
        class_accuracies_after, class_confidences_after = self.calculate_classwise_metrics(global_model_after, testDataloader)

        # Define a threshold for significant changes
        accuracy_change_threshold = 0.1  # 10% change in accuracy
        confidence_change_threshold = 0.1  # 10% change in confidence

        # Find maximum accuracy for scaling the bar length
        max_accuracy = max(
            max([x.item() for x in class_accuracies_client]), 
            max([x.item() for x in class_accuracies_before]), 
            max([x.item() for x in class_accuracies_after])
        )

        # Track whether unlearning has been successful based on accuracy and confidence
        unlearning_successful = False
        affected_classes_count = 0

        # Analyze and interpret the results for each class
        for i in range(num_classes):
            # Extract scalar values using `.item()` method
            accuracy_before = class_accuracies_before[i].item()
            accuracy_after = class_accuracies_after[i].item()
            accuracy_client = class_accuracies_client[i].item()

            confidence_before = class_confidences_before[i].item()
            confidence_after = class_confidences_after[i].item()
            confidence_client = class_confidences_client[i].item()

            print(f"\nClass {i}:")
            
            # Display accuracy and confidence in ASCII bar chart
            print("  Accuracy:")
            print(f"    Client Model:      {self.generate_ascii_bar(accuracy_client, max_accuracy)} ({accuracy_client:.2f})")
            print(f"    Global Before:     {self.generate_ascii_bar(accuracy_before, max_accuracy)} ({accuracy_before:.2f})")
            print(f"    Global After:      {self.generate_ascii_bar(accuracy_after, max_accuracy)} ({accuracy_after:.2f})")

            print("  Confidence:")
            print(f"    Client Model:      {self.generate_ascii_bar(confidence_client, max_accuracy)} ({confidence_client:.2f})")
            print(f"    Global Before:     {self.generate_ascii_bar(confidence_before, max_accuracy)} ({confidence_before:.2f})")
            print(f"    Global After:      {self.generate_ascii_bar(confidence_after, max_accuracy)} ({confidence_after:.2f})")

            # Unlearning criteria: Accuracy/Confidence after should be lower than the client model and closer to the global model before unlearning.
            accuracy_drop_client_to_after = accuracy_client - accuracy_after
            confidence_drop_client_to_after = confidence_client - confidence_after

            # Check for significant drop in accuracy after unlearning compared to client
            if accuracy_drop_client_to_after > accuracy_change_threshold and accuracy_after <= accuracy_before:
                affected_classes_count += 1
                unlearning_successful = True
                print(f"    Significant accuracy drop for Class {i} after unlearning compared to the client model.")

            # Check for significant drop in confidence after unlearning compared to client
            if confidence_drop_client_to_after > confidence_change_threshold and confidence_after <= confidence_before:
                print(f"    Significant confidence drop for Class {i} after unlearning compared to the client model.")
                unlearning_successful = True

            print("---")

        # Final decision on unlearning
        if unlearning_successful:
            print("\nUnlearning has occurred based on class-wise accuracy or confidence drops compared to the client model.")
        else:
            print("\nNo significant accuracy or confidence drops detected compared to the client model. Unlearning may not have been effective.")

    # Function to compute SHAP values using DeepExplainer
    def compute_shap_values(self, model, data_loader, num_samples=50, background_size=10):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        batch = next(iter(data_loader))  # Get a batch of data from the data_loader
        images, labels = batch

        # Ensure images have the correct shape [batch_size, 1, 28, 28] (grayscale images)
        if images.dim() == 5 and images.shape[2] == 1:
            images = images.squeeze(2)  # Remove the extra dimension
        
        if images.shape[1] == 32:  # Indicates the channel dimension is incorrectly set as 32
            # Permute from [batch_size, height, width, channels] to [batch_size, channels, height, width]
            images = images.permute(0, 3, 1, 2)

        # Take a smaller subset for SHAP explanation
        images = images[:num_samples]  # Keep shape as [batch_size, 1, 28, 28]

        # Select a small set of images as background
        background = images[:background_size].to(device)

        # Use SHAP's DeepExplainer
        explainer = shap.DeepExplainer(model, background)

        # Compute SHAP values for the images
        shap_values = explainer.shap_values(images.to(device), check_additivity=False)

        return shap_values, images
    
    # Function to compute mean absolute SHAP values per class
    def compute_mean_shap_per_class(self, shap_values, num_classes):
        mean_shap_per_class = []
        
        # Iterate over each class
        for class_idx in range(num_classes):
            # Compute the mean absolute SHAP value for each class
            class_shap_values = np.abs(shap_values[:, :, :, :, class_idx])  # Take absolute values
            mean_shap_value = np.mean(class_shap_values)
            mean_shap_per_class.append(mean_shap_value)
        
        return mean_shap_per_class
    
    # Function to plot mean feature deviation per class
    def plot_mean_feature_deviation(self, client_idx, test_loader, num_classes=10):
        
        # Load models
        model_before = self.cache.get('initial_model')
        model_before.load_state_dict(self.cache.get('global_models'))
        model_before.eval()

        model_after = self.cache.get('initial_model')
        model_after.load_state_dict(self.cache.get("unlearned_model").state_dict())
        model_after.eval()

        client_model = self.cache.get('initial_model')
        client_model.load_state_dict(self.cache.get("client_models")[client_idx].state_dict())
        client_model.eval()

        # Compute SHAP values for all three models
        shap_values_before, test_images = self.compute_shap_values(model_before, test_loader)
        shap_values_after, _ = self.compute_shap_values(model_after, test_loader)
        shap_values_client, _ = self.compute_shap_values(client_model, test_loader)

        # Compute mean SHAP values per class
        mean_shap_before = self.compute_mean_shap_per_class(shap_values_before, num_classes)
        mean_shap_after = self.compute_mean_shap_per_class(shap_values_after, num_classes)
        mean_shap_client = self.compute_mean_shap_per_class(shap_values_client, num_classes)
        
        # Plot the mean SHAP values for each class
        classes = np.arange(num_classes)
        plt.figure(figsize=(10, 6))
        
        plt.plot(classes, mean_shap_before, label="Global Model Before Unlearning", marker='o')
        plt.plot(classes, mean_shap_after, label="Global Model After Unlearning", marker='s')
        plt.plot(classes, mean_shap_client, label="Client Model", marker='d')
        
        plt.xlabel('Class')
        plt.ylabel('Feature Importance')
        plt.title('Feature Importance Deviation by Class')
        plt.legend()
        plt.grid(True)
        plt.xticks(classes)
        plt.show()
    
    # Function to display SHAP summary as ASCII bar charts
    def display_feature_summary(self, client_idx, test_loader, num_classes=10):
        
        # Load models
        model_before = self.cache.get('initial_model')
        model_before.load_state_dict(self.cache.get('global_models'))
        model_before.eval()

        model_after = self.cache.get('initial_model')
        model_after.load_state_dict(self.cache.get("unlearning_global_models"))
        model_after.eval()

        client_model = self.cache.get('initial_model')
        client_model.load_state_dict(self.cache.get("client_models")[client_idx].state_dict())
        client_model.eval()

        # Compute SHAP values for all three models
        shap_values_before, test_images = self.compute_shap_values(model_before, test_loader)
        shap_values_after, _ = self.compute_shap_values(model_after, test_loader)
        shap_values_client, _ = self.compute_shap_values(client_model, test_loader)

        # Compute mean SHAP values per class
        mean_shap_before = self.compute_mean_shap_per_class(shap_values_before, num_classes)
        mean_shap_after = self.compute_mean_shap_per_class(shap_values_after, num_classes)
        mean_shap_client = self.compute_mean_shap_per_class(shap_values_client, num_classes)

        max_value = max(max(mean_shap_before), max(mean_shap_after), max(mean_shap_client))
        
        # Print ASCII bar chart for each class
        for class_idx in range(num_classes):
            print(f"\nClass {class_idx}:")
            print(f"Client Model {client_idx}:                  {self.generate_ascii_bar(mean_shap_client[class_idx], max_value)} ({mean_shap_client[class_idx]:.4f})")
            print(f"Global Model Before Unlearning:  {self.generate_ascii_bar(mean_shap_before[class_idx], max_value)} ({mean_shap_before[class_idx]:.4f})")
            print(f"Global Model After Unlearning:   {self.generate_ascii_bar(mean_shap_after[class_idx], max_value)} ({mean_shap_after[class_idx]:.4f})")


    def compute_classwise_metrics(self, model, data_loader, num_classes=10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
    
        model.eval()
    
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                
                # If images have shape [N, 1, 1, H, W], remove the extra dimension.
                if images.dim() == 5 and images.size(1) == 1:
                    images = images.squeeze(1)  # now shape becomes [N, 1, H, W]
                
                # If images are 3D (e.g. [N, H, W]), add a channel dimension.
                if images.dim() == 3:
                    images = images.unsqueeze(1)  # now shape becomes [N, 1, H, W]
                
                # If images are in channels-last format (e.g. [N, H, W, 3]) then permute them to [N, 3, H, W].
                if images.dim() == 4 and images.shape[-1] == 3 and images.shape[1] != 3:
                    images = images.permute(0, 3, 1, 2)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
    
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
    
        class_accuracies = [
            class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            for i in range(num_classes)
        ]
        return class_accuracies, class_total




    
    # Function to aggregate the remaining clients' models by averaging their parameters
    def aggregate_remaining_clients(self, client_models):
        
        aggregated_model = copy.deepcopy(client_models[0])  # Use the structure of the first model as the base
        # Initialize the state_dict of the aggregated model with zeros
        for param in aggregated_model.parameters():
            param.data.zero_()

        # Sum the parameters of the remaining client models
        num_clients = len(client_models)
        for client_model in client_models:
            state_dict = client_model.state_dict()
            for param_name, param_tensor in state_dict.items():
                aggregated_model.state_dict()[param_name] += param_tensor

        # Average the parameters
        for param_name in aggregated_model.state_dict():
            aggregated_model.state_dict()[param_name] /= num_clients

        return aggregated_model
    
    # Modified main analysis function to compare client model, remaining clients, and global model
    def display_client_contributions_to_calss_compared_to_others(self, client_idx, testDataloader, num_classes=10, num_clients=10):
        
        # Load models
        model_before = self.cache.get('initial_model')
        model_before.load_state_dict(self.cache.get('global_models'))
        model_before.eval()

        client_model = self.cache.get('initial_model')
        client_model.load_state_dict(self.cache.get("client_models")[client_idx].state_dict())
        client_model.eval()

        # Correct loading of the client models
        remaining_clients_models = []
        for i in range(0, num_clients):  # Assuming client 0 is the one to be unlearned
            if (i != client_idx):
                remainng_client_model = self.cache.get('initial_model')  # Get a fresh instance of the model
                remainng_client_model.load_state_dict(self.cache.get("client_models")[i].state_dict())  # Load the client's state_dict
                remaining_clients_models.append(remainng_client_model)

        # Aggregate the remaining clients' models
        aggregated_remaining_clients_model = self.aggregate_remaining_clients(remaining_clients_models)
        
        # Compute class-wise accuracy for client model
        client_class_accuracies, client_class_totals = self.compute_classwise_metrics(client_model, testDataloader, num_classes)
        
        # Compute class-wise accuracy for aggregated remaining clients' model
        remaining_class_accuracies, remaining_class_totals = self.compute_classwise_metrics(aggregated_remaining_clients_model, testDataloader, num_classes)
        
        # Compute class-wise accuracy for global model
        global_class_accuracies, global_class_totals = self.compute_classwise_metrics(model_before, testDataloader, num_classes)
        
        # # Compute SHAP values for all three models
        max_accuracy = max(
            max(client_class_accuracies), 
            max(remaining_class_accuracies), 
            max(global_class_accuracies)
        )

        # Analyze class-wise impact and feature contributions
        for class_idx in range(num_classes):
            print(f"\nClass {class_idx}:")

            # Class-wise accuracy for each model
            print(f"  Client {client_idx} Model Accuracy:          {self.generate_ascii_bar(client_class_accuracies[class_idx], max_accuracy)} "
                f"({client_class_accuracies[class_idx]:.2f}) with {client_class_totals[class_idx]} samples")

            print(f"  Remaining Clients Model Accuracy: {self.generate_ascii_bar(remaining_class_accuracies[class_idx], max_accuracy)} "
                f"({remaining_class_accuracies[class_idx]:.2f}) with {remaining_class_totals[class_idx]} samples")

            print(f"  Global Model Accuracy:            {self.generate_ascii_bar(global_class_accuracies[class_idx], max_accuracy)} "
                f"({global_class_accuracies[class_idx]:.2f}) with {global_class_totals[class_idx]} samples")

            # Check for potential bias or imbalance
            if client_class_accuracies[class_idx] - remaining_class_accuracies[class_idx] > 0.3:
                print(f"  Warning: Client 0 contributes more to Class {class_idx} than remaining clients.")
            
            if client_class_totals[class_idx] > remaining_class_totals[class_idx]:
                print(f"  Warning: Client 0 has more samples for Class {class_idx}, potential class imbalance after unlearning.")
    
    
    def plot_client_unlearning_impact_bar(self, client_idx, test_loader, num_classes=10, num_clients=10, threshold=0.05):

        # Load global model (before unlearning)
        model_before = self.cache.get('initial_model')
        model_before.load_state_dict(self.cache.get("global_models"))
        model_before.eval()
    
        # Load target client model
        client_model = self.cache.get('initial_model')
        client_model.load_state_dict(self.cache.get("client_models")[client_idx].state_dict())
        client_model.eval()
    
        # Load remaining clients models (all clients except the target)
        remaining_clients_models = []
        for i in range(num_clients):
            if i != client_idx:
                rem_model = self.cache.get('initial_model')  # Get fresh instance
                rem_model.load_state_dict(self.cache.get("client_models")[i].state_dict())
                rem_model.eval()
                remaining_clients_models.append(rem_model)
    
        # Aggregate the remaining client models (using your own method)
        aggregated_remaining_clients_model = self.aggregate_remaining_clients(remaining_clients_models)
    
        # Compute class-wise accuracies using your method.
        client_class_acc, _ = self.compute_classwise_metrics(client_model, test_loader, num_classes)
        remaining_class_acc, _ = self.compute_classwise_metrics(aggregated_remaining_clients_model, test_loader, num_classes)
        global_class_acc, _ = self.compute_classwise_metrics(model_before, test_loader, num_classes)
    
        # For our plot, we'll use the target (client) and aggregated remaining clients.
        target_acc = np.array(client_class_acc)
        remaining_acc = np.array(remaining_class_acc)
        diff = target_acc - remaining_acc
    
        # Define class labels and positions.
        classes = [f'{i}' for i in range(num_classes)]
        ind = np.arange(num_classes)
    
        # Create the bar chart.
        fig, ax = plt.subplots(figsize=(12, 6))
    
        # Color bars red if difference meets/exceeds threshold, otherwise gray.
        bar_colors = ['red' if d >= threshold else 'gray' for d in diff]
        bars = ax.bar(ind, diff, color=bar_colors, edgecolor='black')
    
        # Annotate each bar with the percentage difference.
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.annotate(f'{height*100:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=12)
    
        # Draw a horizontal line at 0 for reference.
        ax.axhline(0, color='black', linewidth=0.8)
    
        # Set axis labels and title.
        ax.set_xticks(ind)
        ax.set_xticklabels(classes, fontsize=14)
        ax.set_xlabel('Classes', fontsize=14)
        ax.set_ylabel('Difference in Accuracy (Target - Remaining)', fontsize=14)
        ax.set_title('Impact of Unlearning: Difference Bar Chart\n(Target Client vs. Aggregated Remaining Clients)', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
    
        # Adjust y-limits to be 0.1 more than the highest and lowest bar.
        y_min = min(diff) - 0.1
        y_max = max(diff) + 0.1
        ax.set_ylim(y_min, y_max)
    
        # Create a custom legend and place it outside the plot.
        legend_elements = [
            Patch(facecolor='red', edgecolor='black', label=f'> {threshold*100:.0f}% difference (Affected)'),
            Patch(facecolor='gray', edgecolor='black', label=f'≤ {threshold*100:.0f}% difference')
        ]
        ax.legend(handles=legend_elements, fontsize=12, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
    
        plt.tight_layout()
        plt.show()


    # Step 1: Modify the function to compute SHAP values for a given class
    def compute_shap_values_for_class(self, model, dataloader, target_class=1, num_samples=100):
        model.eval()  # Set the model to evaluation mode
        
        selected_images = []
        
        # Loop through the dataloader and select samples of the specified class
        for images, labels in dataloader:
            class_indices = (labels == target_class).nonzero(as_tuple=True)[0]
            selected_images.append(images[class_indices])
            
            # Break if we have enough samples
            if len(torch.cat(selected_images)) >= num_samples:
                break

        if len(selected_images) == 0:
            raise ValueError(f"No samples found for class {target_class}.")

        # Concatenate selected images
        selected_images = torch.cat(selected_images)[:num_samples]

        if selected_images.shape[0] == 0:
            raise ValueError(f"Not enough samples for class {target_class}. Found {selected_images.shape[0]} samples.")

        # Use the first few images for background and the rest for testing
        background = selected_images[:20]
        test_images = selected_images[20:23]
        
        # Squeeze the dimensions for SHAP input
        background.squeeze_(1)
        test_images.squeeze_(1)
        
        # SHAP explainer
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(test_images, check_additivity=False)
        
        return shap_values, test_images

    # Step 1: Reshape the input images for visualization
    def reshape_images_for_shap(self, images):
        # Reshape from (N, 28, 28) to (N, 28, 28) - Ensure it's 2D for grayscale
        reshaped_images = images.cpu().numpy().squeeze()  # Remove single dimensions for grayscale images
        return reshaped_images

    def reshape_shap_values(self, shap_values, class_idx=None):
        # If a class is specified, select SHAP values for that class (shape will become (N, 28, 28))
        if class_idx is not None:
            reshaped_shap_values = shap_values[class_idx]  # Class-wise SHAP values
        else:
            # If no class is specified, average SHAP values across all classes
            reshaped_shap_values = np.mean(shap_values, axis=0)  # Shape will become (N, 28, 28)
        
        reshaped_shap_values = reshaped_shap_values.squeeze()  # Ensure 2D (28, 28) shape
        return reshaped_shap_values

    # Step 1: Function to plot SHAP values for three models side-by-side for easy comparison
    def plot_shap_comparison(self, client_idx, testDataloader, sample_index=0):

        # Load models
        model_before = self.cache.get('initial_model')
        model_before.load_state_dict(self.cache.get('global_models'))
        model_before.eval()

        model_after = self.cache.get('initial_model')
        model_after.load_state_dict(self.cache.get("unlearning_global_models"))
        model_after.eval()

        client_model = self.cache.get('initial_model')
        client_model.load_state_dict(self.cache.get("client_models")[client_idx].state_dict())
        client_model.eval()

        # Step 3: Compute and reshape SHAP values
        shap_values_client, test_images = self.compute_shap_values_for_class(client_model, testDataloader, target_class=0)
        shap_values_before, _ = self.compute_shap_values_for_class(model_before, testDataloader, target_class=0)
        shap_values_after, _ = self.compute_shap_values_for_class(model_after, testDataloader, target_class=0)

        images = self.reshape_images_for_shap(test_images)
        shap_values_client = self.reshape_shap_values(shap_values_client)
        shap_values_before = self.reshape_shap_values(shap_values_before)
        shap_values_after = self.reshape_shap_values(shap_values_after)

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))

        # Plot original image
        ax[0].imshow(images[sample_index], cmap='gray')
        ax[0].set_title('Original Image')

        # Plot SHAP values for client model
        shap_client_img = shap_values_client[sample_index]
        ax[1].imshow(shap_client_img, cmap='RdBu', vmin=-np.max(shap_client_img), vmax=np.max(shap_client_img))
        ax[1].set_title('SHAP Client Model')

        # Plot SHAP values for global model before unlearning
        shap_before_img = shap_values_before[sample_index]
        ax[2].imshow(shap_before_img, cmap='RdBu', vmin=-np.max(shap_before_img), vmax=np.max(shap_before_img))
        ax[2].set_title('SHAP Global Model Before Unlearning')

        # Plot SHAP values for global model after unlearning
        shap_after_img = shap_values_after[sample_index]
        ax[3].imshow(shap_after_img, cmap='RdBu', vmin=-np.max(shap_after_img), vmax=np.max(shap_after_img))
        ax[3].set_title('SHAP Global Model After Unlearning')

        plt.show()
    
    # Step 2: Function to plot the difference in SHAP values as a heatmap
    def plot_shap_difference_heatmap(self, client_idx, testDataloader,  sample_index=0):

        # Load models
        model_before = self.cache.get('initial_model')
        model_before.load_state_dict(self.cache.get('global_models'))
        model_before.eval()

        model_after = self.cache.get('initial_model')
        model_after.load_state_dict(self.cache.get("unlearning_global_models"))
        model_after.eval()

        client_model = self.cache.get('initial_model')
        client_model.load_state_dict(self.cache.get("client_models")[client_idx].state_dict())
        client_model.eval()

        shap_values_client, test_images = self.compute_shap_values_for_class(client_model, testDataloader, target_class=0)
        shap_values_before, _ = self.compute_shap_values_for_class(model_before, testDataloader, target_class=0)
        shap_values_after, _ = self.compute_shap_values_for_class(model_after, testDataloader, target_class=0)
        images = self.reshape_images_for_shap(test_images)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Plot original image
        ax[0].imshow(images[sample_index], cmap='gray')
        ax[0].set_title('Original Image')

        # Plot SHAP difference (before vs. after unlearning)
        shap_diff = np.abs(shap_values_before[sample_index] - shap_values_after[sample_index])
        img = ax[1].imshow(shap_diff, cmap='coolwarm', vmin=0, vmax=np.max(shap_diff))
        ax[1].set_title('SHAP Difference (Before vs After)')

        # Highlight the most changed areas with a red box
        ax[2].imshow(images[sample_index], cmap='gray')
        ax[2].imshow(shap_diff, cmap='coolwarm', alpha=0.6)
        ax[2].set_title('Most Changed Areas (Red Overlay)')
        plt.colorbar(img, ax=ax[1])

        plt.show()

    def generate_proxy_data(self, client_idx, data_loader, num_classes=10, num_samples=100, confidence_threshold=0.9):
        """
        Generates proxy data from client model based on high-confidence predictions and prioritizes high-accuracy classes.

        :param client_model: The client model to generate proxy data from.
        :param data_loader: DataLoader for the original dataset.
        :param num_classes: Number of classes in the dataset.
        :param num_samples: Number of samples to generate for the proxy dataset.
        :param confidence_threshold: Confidence threshold for selecting high-confidence samples.
        :return: Concatenated proxy data and proxy labels.
        """
        proxy_data, proxy_labels = [], []
        class_correct = np.zeros(num_classes)  # To track correct predictions per class
        class_total = np.zeros(num_classes)    # To track total predictions per class

        client_model = self.cache.get('initial_model')
        client_model.load_state_dict(self.cache.get("client_models")[client_idx].state_dict())
        client_model.eval()

        client_model.eval()

        # First, calculate class-wise accuracies
        with torch.no_grad():
            for data, labels in data_loader:
                if data.dim() == 5:  # If the image has an extra dimension, squeeze it
                    data = data.squeeze(1)  # Remove the extra dimension

                # Check if the input tensor has the correct shape for CIFAR-10
                if data.shape[1] == 32:  # Indicates the channel dimension is incorrectly set as 32
                    # Permute from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                    data = data.permute(0, 3, 1, 2)
                
                # Get predictions from the client model
                outputs = client_model(data)
                _, predicted = torch.max(outputs, dim=1)
                
                # Update correct and total counts for each class
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1

        # Calculate class accuracies
        class_accuracies = class_correct / class_total
        class_probabilities = class_accuracies / np.sum(class_accuracies)  # Normalize accuracies to get probabilities

        # Generate proxy data by prioritizing high-accuracy classes
        with torch.no_grad():
            for data, _ in data_loader:
                if len(proxy_data) >= num_samples:
                    break
                if data.dim() == 5:  # If the image has an extra dimension, squeeze it
                    data = data.squeeze(1)  # Remove the extra dimension

                # Check if the input tensor has the correct shape for CIFAR-10
                if data.shape[1] == 32:  # Indicates the channel dimension is incorrectly set as 32
                    # Permute from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                    data = data.permute(0, 3, 1, 2)

                # Get predictions and confidence scores from the client model
                outputs = client_model(data)
                probs = F.softmax(outputs, dim=1)  # Get probability distribution
                confidences, predicted = torch.max(probs, dim=1)  # Max confidence and predicted labels

                # Filter based on confidence threshold
                high_confidence_mask = confidences >= confidence_threshold
                if high_confidence_mask.sum() > 0:
                    high_confidence_data = data[high_confidence_mask]
                    high_confidence_labels = predicted[high_confidence_mask]

                    # Prioritize samples from high-accuracy classes
                    for idx, label in enumerate(high_confidence_labels):
                        if len(proxy_data) >= num_samples:
                            break
                        label_class = label.item()
                        
                        # Sample more from high-accuracy classes
                        if np.random.rand() <= class_probabilities[label_class]:
                            # Add to proxy dataset
                            proxy_data.append(high_confidence_data[idx].unsqueeze(0))
                            proxy_labels.append(high_confidence_labels[idx].unsqueeze(0))

                # Stop if we have enough samples
                if len(proxy_data) >= num_samples:
                    break
        
        # Return concatenated proxy data and labels
        return torch.cat(proxy_data), torch.cat(proxy_labels)
    
    # Function to augment data (simple example with noise and flipping)
    def augment_data(self, data):
        noise = torch.randn_like(data) * 0.02  # Add random noise
        augmented_data = data + noise
        
        # Random horizontal flips
        augmented_data = torch.flip(augmented_data, dims=[-1]) if torch.rand(1).item() > 0.9 else augmented_data
        
        return augmented_data

    # Function to evaluate the model on the proxy dataset and calculate accuracy
    def evaluate_on_proxy_data(self, model, proxy_data, proxy_labels):
        model.eval()
        with torch.no_grad():
            outputs = model(proxy_data)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == proxy_labels).sum().item()
        accuracy = correct / len(proxy_labels)
        
        # Optionally, you can also return logits/confidence scores for further analysis
        confidence_scores = torch.softmax(outputs, dim=1).max(dim=1).values
        return accuracy, confidence_scores
    
    def verify_unlearning_effectiveness(self, client_idx, test_loader, num_classes=10, num_samples=100, confidence_threshold=0.9, accuracy_threshold=0.25):
        # Generate proxy data from the client model
        proxy_data, proxy_labels = self.generate_proxy_data(client_idx, test_loader, num_classes, num_samples, confidence_threshold)
        
        # Augment the proxy data
        augmented_data = self.augment_data(proxy_data)
        
        # Load the global model after unlearning
        model_after = self.cache.get('initial_model')
        model_after.load_state_dict(self.cache.get("unlearning_global_models"))
        model_after.eval()

        model_before = self.cache.get('initial_model')
        model_before.load_state_dict(self.cache.get('global_models'))
        model_before.eval()
        
        # Evaluate the model on the augmented proxy data
        accuracy_after, confidence_scores_after = self.evaluate_on_proxy_data(model_after, augmented_data, proxy_labels)
        accuracy_before, confidence_scores_before = self.evaluate_on_proxy_data(model_before, augmented_data, proxy_labels)
        accuracy_drop = accuracy_before - accuracy_after
        
        # Validate unlearning success based on both accuracy and confidence drops
        if accuracy_drop > accuracy_threshold:
            print(f"Unlearning was successful for client {i}. Accuracy dropped by {accuracy_drop:.2f}.")
        else:
            print(f"Unlearning was not successful for client {i}. Accuracy drop: {accuracy_drop:.2f}.")