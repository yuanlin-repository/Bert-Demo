# Your changes here - also print trainable parameters
frozen_alexnet = get_alexnet_model(freeze=True)
print_trainable_params(frozen_alexnet)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(frozen_alexnet.parameters(), lr=0.0005)
frozen_alexnet_result = train_model(frozen_alexnet, train_loader, val_loader, criterion, optimizer, num_epochs=10)
frozen_alexnet_result

# Your graphs here and please provide comment in markdown in another cell
plot_loss_and_accuracy_curves(no_frozen_alexnet_result['train_losses'], 
                              no_frozen_alexnet_result['val_losses'], 
                              no_frozen_alexnet_result['train_accuracies'], 
                              no_frozen_alexnet_result['val_accuracies'], 
                              num_epochs=10, 
                              model_name="No Frozen AlexNet")

plot_loss_and_accuracy_curves(frozen_alexnet_result['train_losses'], 
                              frozen_alexnet_result['val_losses'], 
                              frozen_alexnet_result['train_accuracies'], 
                              frozen_alexnet_result['val_accuracies'], 
                              num_epochs=10, 
                              model_name="Frozen AlexNet")
