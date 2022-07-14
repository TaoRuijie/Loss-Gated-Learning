import os, argparse, pickle, glob
from model import *
from dataLoader import *
from tools import *

parser = argparse.ArgumentParser(description = "Loss Gated Learning")
parser.add_argument('--n_cpu',        type=int, default=8)
parser.add_argument('--max_frames',    type=int, default=300)
parser.add_argument('--batch_size',   type=int, default=512)
parser.add_argument('--init_model',   type=str, default="")
parser.add_argument('--save_path',    type=str, default="")
parser.add_argument('--train_list',   type=str, default="",help='Path for Vox2 list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--val_list',     type=str, default="", help='Path for Vox_O list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--train_path',   type=str, default="", help='Path to the Vox2 set')
parser.add_argument('--val_path',     type=str, default="", help='Path to the Vox_O set')
parser.add_argument('--musan_path',   type=str, default="", help='Path to the musan set')
parser.add_argument('--rir_path',     type=str, default="", help='Path to the rir set')
parser.add_argument('--lr',           type=float, default=0.001)
parser.add_argument('--n_cluster',    type=int, default=6000, help='Number of clusters')
parser.add_argument('--test_interval',type=int, default=1)
parser.add_argument('--max_epoch',    type=int, default=100)
parser.add_argument('--LGL',          dest='LGL', action='store_true', help='Use LGL or baseline only')
args = parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")
inf_max = 10**3
if args.LGL:
	gates = [1, 3, 3, 5, 6] # Set the gates in each iterations, which is different from our paper because we use stronger augmentation in dataloader
else:
	gates = [inf_max, inf_max, inf_max, inf_max, inf_max] # Set the gate as a very large value = No gate (baseline)

args.model_folder = os.path.join(args.save_path, 'model') # Path for the saved models
args.dic_folder   = os.path.join(args.save_path, 'dic') # Path for the saved pseudo label dic
args.score_path   = os.path.join(args.save_path, 'score.txt') # Path for the score file
os.makedirs(args.model_folder, exist_ok = True)
os.makedirs(args.dic_folder, exist_ok = True)
score_file = open(args.score_path, "a+")

stage, best_epoch, next_epoch, iteration = check_clustering(args.score_path, args.LGL) # Check the state of this epoch
print(stage, best_epoch, next_epoch, iteration)

Trainer = trainer(**vars(args)) # Define the framework
modelfiles = glob.glob('%s/model0*.model'%args.model_folder) # Go for all saved model
modelfiles.sort()

if len(modelfiles) >= 1: # Load the previous model
	Trainer.load_parameters(modelfiles[-1])
	args.epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
else:
	args.epoch = 1 # Start from the first epoch
	for items in vars(args): # Save the parameters in args
		score_file.write('%s %s\n'%(items, vars(args)[items]));
	score_file.flush()

if args.epoch == 1:	# Do clustering in the first epoch
	Trainer.load_parameters(args.init_model) # Load the init_model
	clusterLoader = get_Loader(args, cluster_only = True) # Data Loader
	dic_label, NMI = Trainer.cluster_network(loader = clusterLoader, n_cluster = args.n_cluster) # Do clustering
	pickle.dump(dic_label, open(args.dic_folder + "/label%04d.pkl"%args.epoch, "wb")) # Save the pseudo labels
	print_write(type = 'C', text = [args.epoch, NMI], score_file = score_file)
	
labelfiles = glob.glob('%s/label0*.pkl'%args.dic_folder) # Read the last pseudo labels
labelfiles.sort()
dic_label = pickle.load(open(labelfiles[-1], "rb"))
print("Dic %s loaded!"%labelfiles[-1])
trainLoader, clusterLoader = get_Loader(args, dic_label) # data loader with the pseduo labels

while args.epoch <= args.max_epoch:
	stage, best_epoch, next_epoch, iteration = check_clustering(args.score_path, args.LGL) # Check the state of this epoch

	if stage == 'T': # Classification training
		loss, acc, nselects = Trainer.train_network(epoch = args.epoch, loader = trainLoader, gate = inf_max)
		print_write(type = 'T', text = [args.epoch, loss, acc, nselects], score_file = score_file)

	elif stage == 'L': # LGL training
		if best_epoch != None: # LGL start from the best model from 'T' stage
			Trainer.load_parameters('%s/model0%03d.model'%(args.model_folder, best_epoch)) # Load the best model
		loss, acc, nselects = Trainer.train_network(epoch = args.epoch, loader = trainLoader, gate = gates[iteration - 1])
		print_write(type = 'L', text = [args.epoch, loss,  acc, nselects, gates[iteration - 1]], score_file = score_file)

	elif stage == 'C': # Clustering
		iteration += 1
		if iteration > 5: # Maximun iteration is 5
			quit()
		Trainer.load_parameters('%s/model0%03d.model'%(args.model_folder, best_epoch)) # Load the best model
		clusterLoader = get_Loader(args, cluster_only = True) # Cluster loader
		dic_label, NMI = Trainer.cluster_network(loader = clusterLoader, n_cluster = args.n_cluster) # Clustering
		args.epoch = next_epoch
		print_write(type = 'C', text = [args.epoch, NMI], score_file = score_file)
		pickle.dump(dic_label, open(args.dic_folder + "/label%04d.pkl"%args.epoch, "wb")) # Save the pseudo label dic
		print("Dic %s loaded!"%(args.dic_folder + "/label%04d.pkl"%args.epoch))
		Trainer = trainer(**vars(args)) # Define the framework
		Trainer.load_parameters(args.init_model) # Load the init_model
		trainLoader, clusterLoader = get_Loader(args, dic_label) # Get new dataloader with new label dic

	if args.epoch % args.test_interval == 0 and stage != 'C': # evaluation
		Trainer.save_parameters(args.model_folder + "/model%04d.model"%args.epoch) # Save the model
		EER, minDCF = Trainer.eval_network(**vars(args))
		print_write(type = 'E', text = [args.epoch, EER, minDCF], score_file = score_file)

	args.epoch += 1
