import os
from pathlib import Path
import torch
import time
import numpy as np
import copy
import shutil
from tqdm import tqdm
import pandas as pd
import importlib.machinery
import sys

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

class ModelTrainer(object):
	def __init__(self, checkpoint_path='./', verbose = False, epoch_save_freq = 1):

		if not os.path.exists(checkpoint_path):
			Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

		self.checkpoint_path = checkpoint_path

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		self.epoch_save_freq = epoch_save_freq

	def get_next_experiment(self, experiments_path):
		immediate_dirs = [int(folder) for folder in os.listdir(experiments_path) if os.path.isdir(os.path.join(experiments_path, folder))]
		if len(immediate_dirs) == 0:
			return 1

		return max(immediate_dirs) + 1

	def train(self, experiment_number, dataloaders, model, criterion, optimizer, scheduler, start_epoch, end_epoch, continue_training = False):
		self.log_file = os.path.join(self.checkpoint_path, 'log.csv')
		self.experiments_dir = os.path.join(self.checkpoint_path, 'experiments')
		model_file_name = None
		for epoch in range(start_epoch, end_epoch+1):
			print('Epoch {}/{}'.format(epoch, end_epoch))
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
			
				if epoch == start_epoch and phase == 'val':
					current_experiment_dir = os.path.join(self.experiments_dir, str(experiment_number))
					if not os.path.exists(current_experiment_dir):
						Path(current_experiment_dir).mkdir(parents=True, exist_ok=True)

					model_file_name = "model_"+str(int(time.time()))+".py"
					dst = os.path.join(current_experiment_dir, model_file_name)
					if continue_training == False:
						shutil.copy(src="./model.py", dst=dst)
					else:
						shutil.copy(src="./temp_model.py", dst=dst)
					
					if continue_training == False:
						line = "#from,to="+str(start_epoch)+","+str(start_epoch)+"\n"
						line_prepender(dst, line)
					else:
						with open(dst) as f:
							lines = f.readlines()

						lines[0] = "#from,to="+str(start_epoch)+","+str(start_epoch)+"\n"

						with open(dst, "w") as f:
							f.writelines(lines)


				if epoch == start_epoch or epoch % self.epoch_save_freq == 0:
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
				
				if model_file_name is None and phase == 'val':
					print("No variable named model_file_name")
					return
				
				if phase == 'val':
					with open(os.path.join(self.experiments_dir, str(experiment_number), model_file_name)) as f:
						lines = f.readlines()

					new_line = lines[0].rstrip()
					new_line = "#" + str(new_line.split('#')[1].split('=')[0]) + "=" + str(new_line.split('#')[1].split('=')[1].split(',')[0]) + "," + str(epoch)+"\n"
					lines[0] = new_line

					with open(os.path.join(self.experiments_dir, str(experiment_number), model_file_name), "w") as f:
						f.writelines(lines)


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

		self.train(experiment_number = experiment_number, dataloaders = dataloaders, model = model, criterion = criterion, optimizer = optimizer, scheduler = scheduler, start_epoch=1, end_epoch=epochs)
		

		return

	def get_epochs_in_experiment(self, experiment_path):
		immediate_files = [file for file in os.listdir(experiment_path) if os.path.isfile(os.path.join(experiment_path, file))]
		if len(immediate_files) == 0:
			return []

		new_immediate_files = []
		for el in immediate_files:
			if el.split(".")[0].split("_")[0] == 'epoch':
				new_immediate_files.append(int(el.split(".")[0].split("_")[1]))
		return new_immediate_files

	def copy_experiment_till_epoch(self, experiment, new_experiment, till_epoch):
		experiment_dir = os.path.join(self.checkpoint_path, 'experiments', str(experiment))
		new_experiment_dir = os.path.join(self.checkpoint_path, 'experiments', str(new_experiment))

		if not os.path.exists(new_experiment_dir):
			Path(new_experiment_dir).mkdir(parents=True, exist_ok=True)

		immediate_files = [file for file in os.listdir(experiment_dir) if os.path.isfile(os.path.join(experiment_dir, file))]

		for el in immediate_files:
			if el.split(".")[0].split("_")[0] == 'epoch':
				if int(el.split(".")[0].split("_")[1]) >= till_epoch:
					continue
				else:
					src = os.path.join(experiment_dir, el)
					shutil.copy(src=src, dst=new_experiment_dir)

			if el.split(".")[0].split("_")[0] == 'model':
				model_dir = os.path.join(experiment_dir, el)

				with open(model_dir) as f:
					lines = f.readlines()

				new_line = lines[0].rstrip()

				model_start_epoch = int(new_line.split('#')[1].split('=')[1].split(',')[0])
				model_end_epoch = int(new_line.split('#')[1].split('=')[1].split(',')[1])

				if model_end_epoch <= till_epoch - 1:
					src = os.path.join(experiment_dir, el)
					shutil.copy(src=src, dst=new_experiment_dir)

				else:
					if till_epoch - 1 >= model_start_epoch:
						# copy and update first line
						src = os.path.join(experiment_dir, el)
						shutil.copy(src=src, dst=new_experiment_dir)

						with open(os.path.join(new_experiment_dir, el)) as f:
							lines = f.readlines()

						new_line = lines[0].rstrip()
						new_line = "#" + str(new_line.split('#')[1].split('=')[0]) + "=" + str(new_line.split('#')[1].split('=')[1].split(',')[0]) + "," + str(till_epoch - 1)+"\n"
						lines[0] = new_line

						with open(os.path.join(new_experiment_dir, el), "w") as f:
							f.writelines(lines)
							
					else:
						continue



		df = pd.read_csv(os.path.join(self.checkpoint_path, 'log.csv'))

		df2 = df[(df['experiment_no'] == experiment) & (df['epoch'] < till_epoch)].copy()

		df2['experiment_no'] = [new_experiment] * len(df2)
		df = df.append(df2, ignore_index=True)
		df.to_csv(os.path.join(self.checkpoint_path, 'log.csv'), header=True, index=False)
	
	def get_latest_model_file_name_in_experiment(self, experiment_number):
		experiment_dir = os.path.join(self.checkpoint_path, 'experiments', str(experiment_number))
		immediate_files = [file for file in os.listdir(experiment_dir) if os.path.isfile(os.path.join(experiment_dir, file))]
		
		latest = 0
		latest_file_name = ""
		for el in immediate_files:
			if el.split(".")[0].split("_")[0] == 'model':
				time = int(el.split(".")[0].split("_")[1])
				if time > latest:
					latest = time
					latest_file_name = el

		return latest_file_name
		

	def continue_training(self, dataloaders, experiment_number = 1, from_epoch = 6, update_optimizer = False, till_epoch=10):
		package = "os"
		name = "path"

		imported = getattr(__import__(package, fromlist=[name]), name)

		experiment_dir = os.path.join(self.checkpoint_path, 'experiments', str(experiment_number))
		if not os.path.isdir(experiment_dir):
			print("No such experiment exists")
			return

		saved_epochs = self.get_epochs_in_experiment(experiment_dir)

		possible_start_epochs = [x+1 for x in saved_epochs]
		if from_epoch not in possible_start_epochs:
			print("Possible start epochs are only : " + str(possible_start_epochs))
			return

		if from_epoch != max(possible_start_epochs):
			experiment_number_new = self.get_next_experiment(os.path.join(self.checkpoint_path, 'experiments'))
			# print(experiment_number, experiment_number_new)
			print("A new experiment will be created ("+str(experiment_number_new)+") with all checkpoints and logs till epoch " + str(from_epoch - 1))

			self.copy_experiment_till_epoch(experiment=experiment_number, new_experiment=experiment_number_new, till_epoch=from_epoch)

		else:
			experiment_number_new = experiment_number

		if update_optimizer == False:
			latest_model_file_name = self.get_latest_model_file_name_in_experiment(experiment_number_new)
			print("Will use the latest model file, i.e., " + str(latest_model_file_name))

			latest_model_file = os.path.join(self.checkpoint_path, 'experiments', str(experiment_number_new), latest_model_file_name)

			shutil.copy(src=latest_model_file, dst="./temp_model.py")

			from temp_model import model, criterion, optimizer

			latest_epoch_file = os.path.join(self.checkpoint_path, 'experiments', str(experiment_number_new), "epoch_"+str(from_epoch - 1)+".pt")
			checkpoint = torch.load(latest_epoch_file)
			model.load_state_dict(checkpoint['model'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			scheduler = None

			self.train(experiment_number = experiment_number_new, dataloaders = dataloaders, model = model, criterion = criterion, optimizer = optimizer, scheduler = scheduler, start_epoch=from_epoch, end_epoch=till_epoch, continue_training=True)

			os.remove("./temp_model.py")

		else:

			latest_model_file_name = self.get_latest_model_file_name_in_experiment(experiment_number_new)
			print("Will use the latest model file, i.e., " + str(latest_model_file_name))

			latest_model_file = os.path.join(self.checkpoint_path, 'experiments', str(experiment_number_new), latest_model_file_name)

			shutil.copy(src=latest_model_file, dst="./temp_model.py")

			print('Do you simply want to change the optimizer paramters (type yes) or the entire optimizer (type no)?')
			answer = input()

			if answer.lower() == 'yes' or answer.lower() == 'y':
				print("Update the temp_model.py file with the updated parameters. NOTE: Do not change the optimizer. type 'Done' after completion.")
				answer2 = input()
				if answer2.lower() != "done":
					print("Quitting")
					return

				from temp_model import model, criterion, optimizer
				latest_epoch_file = os.path.join(self.checkpoint_path, 'experiments', str(experiment_number_new), "epoch_"+str(from_epoch - 1)+".pt")
				checkpoint = torch.load(latest_epoch_file)
				model.load_state_dict(checkpoint['model'])
				optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
				scheduler = None

				self.train(experiment_number = experiment_number_new, dataloaders = dataloaders, model = model, criterion = criterion, optimizer = optimizer, scheduler = scheduler, start_epoch=from_epoch, end_epoch=till_epoch, continue_training=True)

				os.remove("./temp_model.py")
			else:
				print("Update the temp_model.py file with the new optimizer. type 'Done' after completion.")
				answer2 = input()
				if answer2.lower() != "done":
					print("Quitting")
					return

				from temp_model import model, criterion, optimizer
				latest_epoch_file = os.path.join(self.checkpoint_path, 'experiments', str(experiment_number_new), "epoch_"+str(from_epoch - 1)+".pt")
				checkpoint = torch.load(latest_epoch_file)
				model.load_state_dict(checkpoint['model'])
				scheduler = None

				self.train(experiment_number = experiment_number_new, dataloaders = dataloaders, model = model, criterion = criterion, optimizer = optimizer, scheduler = scheduler, start_epoch=from_epoch, end_epoch=till_epoch, continue_training=True)

				os.remove("./temp_model.py")
				
		return
