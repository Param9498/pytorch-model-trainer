import os
from pathlib import Path
import torch
import time
import numpy as np
import copy
import shutil

class ModelTrainer(object):
    def __init__(self, checkpoint_path='./', verbose = False, epoch_save_freq = 1):

        if not os.path.exists(checkpoint_path):
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = checkpoint_path

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.epoch_save_freq = epoch_save_freq

    def get_next_experiment(self, experiment_path):
        immediate_dirs = [int(folder) for folder in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, folder))]
        if len(immediate_dirs) == 0:
            return 1

        return max(immediate_dirs) + 1

    def train_model(self, dataloaders, model, criterion, optimizer, scheduler = None, epochs = 25):
        # model = model.to(self.device)

        self.log_file = os.path.join(self.checkpoint_path, 'log.csv')
        if not os.path.exists(self.log_file):
            Path(self.log_file).touch()
            log_file = open(self.log_file, 'a+')
            log_file.write('experiment_no,epoch,is_model_saved,time_taken,phase,loss,metric\n')
            log_file.close()


        self.experiments_dir = os.path.join(self.checkpoint_path, 'experiments')
        if not os.path.exists(self.experiments_dir):
            Path(self.experiments_dir).mkdir(parents=True, exist_ok=True)

        experiment_number = self.get_next_experiment(self.experiments_dir)


        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 10)
            np.random.seed()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                start = time.time()
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                n_samples = 0

                # Iterate over data.
                for _, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    batchSize = inputs.size(0)
                    n_samples += batchSize

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            if scheduler is not None:
                                scheduler.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    pred_top_1 = torch.topk(outputs, k=1, dim=1)[1]
                    running_corrects += pred_top_1.eq(labels.view_as(pred_top_1)).int().sum().item()								

                # Metrics
                top_1_acc = running_corrects / n_samples
                epoch_loss = running_loss / n_samples

                time_elapsed = time.time() - start

                print('{} phase -> loss: {:.4f}, accuracy: {:.4f} \n'.format(phase, epoch_loss, top_1_acc))

                if (epoch + 1) == 1:
                    current_experiment_dir = os.path.join(self.experiments_dir, str(experiment_number))
                    if not os.path.exists(current_experiment_dir):
                        Path(current_experiment_dir).mkdir(parents=True, exist_ok=True)

                    shutil.copy(src="./model.py", dst=current_experiment_dir)

                if (epoch + 1) == 1 or (epoch + 1) % self.epoch_save_freq == 0:
                    log_file = open(self.log_file, 'a+')
                    log_file.write(f'{experiment_number},{epoch},True,{time_elapsed},{phase},{epoch_loss},{top_1_acc}\n')
                    log_file.close()

                    if phase == 'val':
                        checkpoint = { 
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'criterion': criterion,
                            'scheduler': scheduler
                        }

                        torch.save(checkpoint, '%s/epoch_%d.pt' % (current_experiment_dir, epoch))
                else:
                    log_file = open(self.log_file, 'a+')
                    log_file.write(f'{experiment_number},{epoch},False,{time_elapsed},{phase},{epoch_loss},{top_1_acc}\n')
                    log_file.close()

        return
