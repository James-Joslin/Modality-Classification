import model_architectures
import util
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import training_config
import datetime
from imblearn.combine import SMOTETomek
from collections import Counter

if __name__ == "__main__":
    multi_classifier_config = training_config.bracketClassifierConfig()
    util.set_seed(multi_classifier_config.SEED) 
    
    # Personal info - secured by a secrets .json file kept out of the repo by the .gitignore
    with open('secrets.json', 'r') as file:
        secrets = json.load(file)
    file.close()
    connection_params = {
        "username": secrets['sqlConfig']['username'],
        "password": secrets['sqlConfig']['password'],
        "port": secrets['sqlConfig']['port'],
        "node_ip": secrets['sqlConfig']['ip'],
        "db_name": secrets['sqlConfig']['db_name']
    }
    
    ## 2 stage prediction - predicting "last" core10 and PGSI from "first" values and referral categories to be used in the final classifier
    # pgci core 10 and referral categories encoded numerically with an index
    base_training_data = util.get_data(**connection_params, 
                                       query_path=secrets["sqlQueryFilePaths"]["training_data"])  # pandas dataframe via sql query 
    # encoder modality and send values to lookup up table in sql server to provide a cleaner response from the final model outcomes
    modality_encoder = util.InitialModalityEncoder('modality_type')
    modality_encoder.fit(base_training_data)
    base_training_data = modality_encoder.transform(base_training_data)
    # Get the lookup table
    modality_lookup_table = modality_encoder.get_lookup_table()
    print(modality_lookup_table)
    print(base_training_data)
    util.write_to_sql(**connection_params, table_name=secrets['sqlConfig']['initial_mod_lookup'], excel_file=None, dataframe=modality_lookup_table)
    
    # get minimax values to minmax transform pgsi and core10
    minimax_df = util.get_data(**connection_params, 
                               query_path=secrets["sqlQueryFilePaths"]["minimax"])
    print(base_training_data.info())
    
    # Create training sets
    X_data = base_training_data[['unique_referral_index', 'first_pgsi', 'first_core10']].copy()
    y_data = base_training_data[['last_max_pgsi_bracket', 'last_max_core10_bracket']].copy()
    
    # minmax transform transform first pgsi and core10 values
    X_data['first_pgsi'] = X_data['first_pgsi']/minimax_df[minimax_df['score_type'] == 'PGSI']['max_score'].iloc[0].astype('float32')
    X_data['first_core10'] = X_data['first_core10']/minimax_df[minimax_df['score_type'] == 'CORE10']['max_score'].iloc[0].astype('float32')
    
    # One-hot encoding the 'referral_source_index' column
    referral_one_hot_encoded, _ = util.one_hot_encode(base_training_data[['unique_referral_index']].to_numpy().flatten()-1) # subtract 1 as indices are not 0 indexed
    referral_one_hot_encoded = pd.DataFrame(referral_one_hot_encoded, 
                                    columns=[f"referral_source_{int(i+1)}" for i in range(referral_one_hot_encoded.shape[1])])
    X_data = pd.concat([X_data.drop('unique_referral_index', axis=1), referral_one_hot_encoded], axis=1)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = util.train_val_test_split(X_data.to_numpy().astype('float32'), y_data.to_numpy().astype('float32'))
    
    # train tensors
    train_inputs = torch.tensor(X_train, dtype=torch.float32)
    train_targets1 = torch.tensor(Y_train[:, :1].flatten(), dtype=torch.long)
    train_targets2 = torch.tensor(Y_train[:, 1:].flatten(), dtype=torch.long)
    # val tensors
    val_inputs = torch.tensor(X_val, dtype=torch.float32)
    val_targets1 = torch.tensor(Y_val[:, :1].flatten(), dtype=torch.long)
    val_targets2 = torch.tensor(Y_val[:, 1:].flatten(), dtype=torch.long)
    # test tensors
    test_inputs = torch.tensor(X_test, dtype=torch.float32)
    test_targets1 = torch.tensor(Y_test[:, :1].flatten(), dtype=torch.long)
    test_targets2 = torch.tensor(Y_test[:, 1:].flatten(), dtype=torch.long)
    
    # Finalise datasets
    train_dataset = TensorDataset(train_inputs, train_targets1, train_targets2)
    val_dataset = TensorDataset(val_inputs, val_targets1, val_targets2)
    test_dataset = TensorDataset(test_inputs, test_targets1, test_targets2)

    # Build data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=multi_classifier_config.BATCH, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=multi_classifier_config.BATCH, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=multi_classifier_config.BATCH, shuffle=True)

    # intialised model to predict last pgsi and core10 brackets
    metric_classifier = model_architectures.multiOutputNet(
        size_in=X_train.shape[1], 
        hidden_sizes_shared=multi_classifier_config.HIDDEN_SIZES_SHARED, 
        hidden_sizes_a=multi_classifier_config.HIDDEN_SIZES_A, 
        hidden_sizes_b=multi_classifier_config.HIDDEN_SIZES_B, 
        size_out1=Y_train[:, :1].max()+1, 
        size_out2=Y_train[:, 1:].max()+1, 
        dropout_p=multi_classifier_config.DROPOUT_RATE
    ).to(multi_classifier_config.DEVICE)
    # metric_classifier.apply(model_architectures.initialize_weights) # utilise xavier uniform initialised weights
    util.model_summary(metric_classifier, (1,X_train.shape[1]))
    
    if multi_classifier_config.TRAIN:
        optimizer = torch.optim.Adam(
            metric_classifier.parameters(), 
            lr=multi_classifier_config.LR,
            weight_decay=multi_classifier_config.WEIGHT_DECAY,
            # momentum=0.9
        )
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        class_frequencies = util.get_class_frequencies_multi_target(train_loader, 4, 5)
        class_weights1 = util.calculate_alpha(class_frequencies[0]) **multi_classifier_config.WEIGHTING_POWER_A
        class_weights2 = util.calculate_alpha(class_frequencies[1]) **multi_classifier_config.WEIGHTING_POWER_B
        criterion1 = nn.CrossEntropyLoss(weight= class_weights1.to(multi_classifier_config.DEVICE))
        criterion2 = nn.CrossEntropyLoss(weight= class_weights2.to(multi_classifier_config.DEVICE))
        
        if multi_classifier_config.RELOAD:
            checkpoint = util.load_checkpoint(
                util.find_latest_file(multi_classifier_config.BASE_PATH)
            )
            metric_classifier.load_state_dict(
                checkpoint['model_state_dict']
            )
        
        best_val_loss = float('inf')
        l1_lambda = multi_classifier_config.L1_LAMBDA
        for epoch in range(multi_classifier_config.EPOCHS):
            # Training Phase
            running_loss = 0
            total_batches = len(train_loader)
            metric_classifier.train() 
            for i, data_batch in enumerate(train_loader):
                inputs, targets1, targets2 = data_batch
                
                if multi_classifier_config.USE_NOISE:
                    alpha = [multi_classifier_config.DIRICHLET_ALPHA] * inputs.shape[1]  # Adjust alpha as needed
                    noise_ratio = multi_classifier_config.DIRICHLET_RATIO  # Define or vary this value
                    inputs = util.add_dirichlet_noise(inputs, alpha, noise_ratio) # generates alpah noise and reshapes between -1 and 1
                    
                optimizer.zero_grad()

                # Forward pass
                outputs1, outputs2 = metric_classifier(inputs.to(multi_classifier_config.DEVICE))
                l1_reg = sum(param.abs().sum() for param in metric_classifier.parameters())

                
                loss1 = criterion1(outputs1, targets1.to(multi_classifier_config.DEVICE))
                loss2 = criterion2(outputs2, targets2.to(multi_classifier_config.DEVICE))
                total_loss = loss1 + loss2 # + l1_lambda * l1_reg
                
        
                # loss = criterion([outputs1, outputs2], [targets1.to(multi_classifier_config.DEVICE), targets2.to(multi_classifier_config.DEVICE)])
                # total_loss = loss + l1_lambda * l1_reg

                total_loss.backward()
                optimizer.step()
                # scheduler.step()
                running_loss += total_loss.item()
                
            average_loss = running_loss / total_batches
            print(f"Epoch {epoch+1}, Average Loss: {average_loss}")
            
            average_val_loss = util.evaluate_model(metric_classifier, val_loader, (criterion1,criterion2), l1_lambda, l1_reg, multi_classifier_config.DEVICE)
            print(f"Epoch {epoch+1}, Average Validation Loss: {average_val_loss}")
            
            if average_val_loss < best_val_loss:
                util.save_checkpoint(epoch+1, metric_classifier, optimizer, average_val_loss, multi_classifier_config.CHECKPOINT)
                best_val_loss = average_val_loss
    
    # test multi classfier for last pgsi and core10 bracket prediction
    checkpoint = util.load_checkpoint(
        util.find_latest_file(multi_classifier_config.BASE_PATH)
    )
    metric_classifier.load_state_dict(
        checkpoint['model_state_dict']
    )
    
    if multi_classifier_config.TEST:
        metric_classifier.eval()
        all_results1 = np.array([])
        all_results2 = np.array([])
        all_targets1 = np.array([])
        all_targets2 = np.array([])
        for i, data_batch in enumerate(test_loader):
            inputs, targets1, targets2 = data_batch
            outputs1, outputs2 = metric_classifier(
                inputs.to(multi_classifier_config.DEVICE)
            )
            print(outputs1)
            print(outputs2)
            outputs1 = torch.argmax(outputs1.detach().to('cpu'), dim = 1).numpy().flatten()
            outputs2 = torch.argmax(outputs2.detach().to('cpu'), dim = 1).numpy().flatten()
            targets1 = targets1.detach().to('cpu').numpy().flatten()
            targets2 = targets2.detach().to('cpu').numpy().flatten()
        
            # Append the outputs and targets to respective arrays
            all_results1 = np.append(all_results1, outputs1)
            all_results2 = np.append(all_results2, outputs2)
            all_targets1 = np.append(all_targets1, targets1)
            all_targets2 = np.append(all_targets2, targets2)
        
        accuracy1 = np.mean(all_results1 == all_targets1)
        accuracy2 = np.mean(all_results2 == all_targets2)
        print(f"Accuracy for Last PGSI Brackets: {(accuracy1*100):.2f}%")
        print(f"Accuracy for Last Core10 Brackets: {(accuracy2*100):.2f}%")
    
    # Training for modality model
    # Modality training config65    
    modality_classifier_config = training_config.modalityClassifierConfig()
    
    # Build policy vectors and combine with x data of final network to predict modality
    # use original x data in its entirety
    
    X_data = X_data.to_numpy()
    y_data = base_training_data[['first_element_int']].to_numpy()
   
    X_inputs = torch.tensor(X_data, dtype=torch.float32, requires_grad=False)
    X_dataset = TensorDataset(X_inputs)
    X_loader = DataLoader(dataset=X_dataset, batch_size=multi_classifier_config.BATCH, shuffle=False)
    
    vectors1 = None
    vectors2 = None
    
    with torch.no_grad():
        for data_batch in X_loader:
            # print(data_batch)
            outputs1, outputs2 = metric_classifier(
                data_batch[0].to(multi_classifier_config.DEVICE)
            )
            # print(outputs1)
            outputs1 = outputs1.detach().to('cpu').numpy()
            outputs2 = outputs2.detach().to('cpu').numpy()

            # Append the outputs and targets to respective arrays
            if vectors1 is not None:
                vectors1 = np.append(vectors1, outputs1, axis = 0)
                vectors2 = np.append(vectors2, outputs2, axis = 0)
            else:
                vectors1 = outputs1
                vectors2 = outputs2
        
    combined_vectors = np.concatenate([vectors1, vectors2], axis=1)
    
    X_data = base_training_data[['first_pgsi', 'first_core10']].copy()
    X_data['first_pgsi'] = X_data['first_pgsi']/minimax_df[minimax_df['score_type'] == 'PGSI']['max_score'].iloc[0].astype('float32')
    X_data['first_core10'] = X_data['first_core10']/minimax_df[minimax_df['score_type'] == 'CORE10']['max_score'].iloc[0].astype('float32')
    
    final_x_data = np.concatenate([X_data.to_numpy(), referral_one_hot_encoded.to_numpy()], axis=1)
    final_x_data = np.concatenate([final_x_data, combined_vectors], axis=1)
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(final_x_data, y_data)

    # move bracket model back to cpu and of gpu
    metric_classifier = metric_classifier.to('cpu')
    
    
    # y_data, out_size = util.one_hot_encode(y_data.flatten())
    X_train, X_val, X_test, Y_train, Y_val, Y_test = util.train_val_test_split(X_res.astype('float32'), y_res.flatten().astype('float32'), )
    # util.plot_class_histograms(
    #     Y_train
    # )
    # util.plot_class_histograms(
    #     Y_val
    # )
    # util.plot_class_histograms(
    #     Y_test
    # )
    
    # train tensors
    train_inputs = torch.tensor(X_train, dtype=torch.float32)
    train_targets = torch.tensor(Y_train, dtype=torch.long)
    # val tensors
    val_inputs = torch.tensor(X_val, dtype=torch.float32)
    val_targets = torch.tensor(Y_val, dtype=torch.long)
    # test tensors
    test_inputs = torch.tensor(X_test, dtype=torch.float32)
    test_targets = torch.tensor(Y_test, dtype=torch.long)
    
    # Finalise datasets
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)

    # Build data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=modality_classifier_config.BATCH, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=modality_classifier_config.BATCH, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=modality_classifier_config.BATCH, shuffle=False)
    
    modality_classifier = model_architectures.ModalityTypeClassifier(
        size_in=final_x_data.shape[1],
        num_hidden=modality_classifier_config.NUM_HIDDEN,
        hidden_size=modality_classifier_config.HIDDEN_SIZE,
        size_out=y_res.max()+1,
        dropout_p=modality_classifier_config.DROPOUT_RATE
    ).to(modality_classifier_config.DEVICE)
    modality_classifier.apply(model_architectures.initialize_weights)
    util.model_summary(modality_classifier, (1,final_x_data.shape[1]))
    
    if modality_classifier_config.TRAIN:
        modality_optimizer = torch.optim.Adam(modality_classifier.parameters(), lr=modality_classifier_config.LR, weight_decay=modality_classifier_config.WEIGHT_DECAY)
        class_counts = np.zeros(3)
        # Iterate over the data loader
        for target in y_res:  # used startified splitting therefore train, val and test will be the same
            class_counts[target] += 1
        class_counts = torch.Tensor(class_counts)
        class_weights = util.calculate_alpha(class_counts) ** modality_classifier_config.WEIGHTING_POWER
        mod_criterion = nn.CrossEntropyLoss(weight= class_weights.to(modality_classifier_config.DEVICE))
        
        if modality_classifier_config.RELOAD:
            checkpoint = util.load_checkpoint(
                util.find_latest_file(modality_classifier_config.BASE_PATH)
            )
            modality_classifier.load_state_dict(
                checkpoint['model_state_dict']
            )
        
        best_val_loss = float('inf')
        l1_lambda = modality_classifier_config.L1_LAMBDA
        for epoch in range(modality_classifier_config.EPOCHS):
            # Training Phase
            running_loss = 0
            total_batches = len(train_loader)
            modality_classifier.train() 
            for i, data_batch in enumerate(train_loader):
                inputs, modalities = data_batch
                
                if modality_classifier_config.USE_NOISE:
                    alpha = [modality_classifier_config.DIRICHLET_ALPHA] * inputs.shape[1]  # Adjust alpha as needed
                    noise_ratio = modality_classifier_config.DIRICHLET_RATIO  # Define or vary this value
                    inputs = util.add_dirichlet_noise(inputs, alpha, noise_ratio) # generates alpah noise and reshapes between -1 and 1
                
                modality_optimizer.zero_grad()

                # Forward pass
                predictions = modality_classifier(inputs.to(modality_classifier_config.DEVICE))
                l1_reg = sum(param.abs().sum() for param in modality_classifier.parameters()) # not used

                loss = mod_criterion(predictions, modalities.to(modality_classifier_config.DEVICE))
                total_loss = loss # + l1_lambda * l1_reg
                
                loss.backward()
                modality_optimizer.step()
                running_loss += loss
                
            average_loss = running_loss / total_batches
            print(f"Epoch {epoch+1}, Average Loss: {average_loss}")
            
            average_val_loss = util.evaluate_modality_model(modality_classifier, val_loader, mod_criterion, l1_lambda, l1_reg, modality_classifier_config.DEVICE)
            print(f"Epoch {epoch+1}, Average Validation Loss: {average_val_loss}")
            
            if average_val_loss < best_val_loss:
                util.save_checkpoint(epoch+1, modality_classifier, modality_optimizer, average_val_loss, modality_classifier_config.CHECKPOINT)
                best_val_loss = average_val_loss

    if modality_classifier_config.TEST:
        modality_classifier.eval()
        all_results = np.array([])
        all_targets = np.array([])
        for i, data_batch in enumerate(test_loader):
            inputs, targets = data_batch
            # print(len(inputs), len(targets))
            outputs = modality_classifier(
                inputs.to(modality_classifier_config.DEVICE)
            )
            outputs = torch.argmax(outputs.detach().to('cpu'), dim = 1).numpy().flatten()
            targets = targets.detach().to('cpu').numpy().flatten()
            # print(len(targets))
        
            # Append the outputs and targets to respective arrays
            all_results = np.append(all_results, outputs)
            all_targets = np.append(all_targets, targets)
            
        accuracy = np.mean(all_results == all_targets)
        print(f"Accuracy for Modality Classifier: {accuracy}")