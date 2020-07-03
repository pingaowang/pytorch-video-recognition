import os

import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

from utils.general import *

from utils.roc_score import multiclass_roc_score

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model

from exp_config_reader import *
import torch.backends.cudnn as cudnn
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

clip_len = CLIP_LEN
grayscale = GRAY_SCALE

nEpochs = MAX_EPOCH  # Number of epochs for training
BS = BATCH_SIZE  # batch size
resume_epoch = RESUM_EPOCH  # Default is 0, change if want to resume
resume_model_path = RESUM_MODEL_PATH
useTest = USE_TEST  # See evolution of the test set when training
nTestInterval = N_TEST_INTERVAL # Run on test set every nTestInterval epochs
snapshot = SNAPSHOT  # Store a model every snapshot epochs

lr = INIT_LEARNING_RATE # Learning rate

IF_PREPROCESS_TRAIN = False
IF_PREPROCESS_VAL = False
IF_PREPROCESS_TEST = False

dataset = DATASET # Options: hmdb51 or ucf101
_optimizer = OPTIMIZER

if dataset == 'hmdb51':
    num_classes = 51
elif dataset == 'ucf101':
    num_classes = N_CLASSES
elif dataset == 'ucf_motion':
    num_classes = N_CLASSES
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

# save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))


"""
Folder Preparing
"""
# check and make log folder
if not os.path.isdir(LOG_ROOT):
    os.mkdir(LOG_ROOT)
assert not os.path.isdir(LOG_PATH), "The log folder exists."
# check and make saved_models folder
if not os.path.isdir(SAVE_ROOT):
    os.mkdir(SAVE_ROOT)
# save folders
SAVE_FILE_FOLDER = os.path.join(SAVE_ROOT, EXP_NAME)
# create saving folders
assert not os.path.isdir(SAVE_FILE_FOLDER), "The model saving folder exists."
os.mkdir(SAVE_FILE_FOLDER)


# exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

# Need to be removed.
# if resume_epoch != 0:
#     runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
#     run_id = int(runs[-1].split('_')[-1]) if runs else 0
# else:
#     runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
#     run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

modelName = MODEL_NAME  # Options: C3D or R2Plus1D or R3D

def train_model(dataset=dataset, save_dir=SAVE_FILE_FOLDER, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=IF_PRETRAIN)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    if _optimizer == "SGD":
        optimizer = optim.SGD(train_params, lr=lr, momentum=MOMENTUM, weight_decay=WD)
    elif _optimizer == "Adam":
        optimizer = optim.Adam(train_params, lr=lr, weight_decay=WD)
    # print(optimizer)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE,
                                          gamma=SCHEDULER_GAMMA)  # the scheduler divides the lr by 10 every 10 epochs

    model.to(device)
    criterion.to(device)

    # if resume_epoch == 0:
    if resume_model_path == None:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            resume_model_path,
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(resume_model_path))
        model.load_state_dict(checkpoint['state_dict'])
        if RESUM_OPTIMIZER:
            optimizer.load_state_dict(checkpoint['opt_dict'])
        # resume_epoch
    # else:
    #     checkpoint = torch.load(os.path.join(SAVE_FILE_FOLDER, 'models', EXP_NAME + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
    #                             map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
    #     print("Initializing weights from: {}...".format(
    #         os.path.join(SAVE_FILE_FOLDER, EXP_NAME + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    writer = SummaryWriter(logdir=LOG_PATH)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=clip_len, preprocess=IF_PREPROCESS_TRAIN, grayscale=grayscale), batch_size=BS, shuffle=True, num_workers=N_WORKERS)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=clip_len, preprocess=IF_PREPROCESS_VAL, grayscale=grayscale), batch_size=BS, num_workers=N_WORKERS)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=clip_len, preprocess=IF_PREPROCESS_TEST, grayscale=grayscale), batch_size=BS, num_workers=N_WORKERS)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    cudnn.benchmark = True

    global_best_val_acc = 0

    for epoch in range(num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0
            # running_roc = 0.0

            list_pred = list()
            list_label = list()

            # print(optimizer)

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            # for inputs, labels in tqdm(trainval_loaders[phase]):
            run_count = 0
            for inputs, labels in trainval_loaders[phase]:
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # try:
                #     running_roc += roc_auc_score(labels.data.cpu(), preds.cpu())
                # except:
                #     y_true = labels.data.cpu().tolist()
                #     y_true_2 = y_true.copy()
                #     for i_cls in range(N_CLASSES):
                #         y_true_2.append(i_cls)
                #
                #     y_pred = preds.cpu().tolist()
                #     y_pred_2 = y_pred.copy()
                #     for i_cls in range(N_CLASSES):
                #         y_pred_2.append(i_cls)
                #
                #     running_roc += roc_auc_score(y_true_2, y_pred_2)
                #
                # run_count += 1
                list_label += labels.data.cpu().tolist()
                list_pred += preds.cpu().tolist()

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]
            epoch_roc = multiclass_roc_score(label=list_label, pred=list_pred, n_cls=N_CLASSES)

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
                writer.add_scalar('data/train_roc_epoch', epoch_roc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)
                writer.add_scalar('data/val_roc_epoch', epoch_roc, epoch)
                # if epoch_acc >= global_best_val_acc:
                #     torch.save({
                #         'epoch': epoch + 1,
                #         'state_dict': model.state_dict(),
                #         'opt_dict': optimizer.state_dict(),
                #     }, os.path.join(SAVE_FILE_FOLDER, 'models', EXP_NAME + '_epoch-' + str(epoch) + 'ValAcc_{:10.4f}_'.format(epoch_loss) + '.pth.tar'))
                #     print("Save model at {}\n".format(
                #         os.path.join(SAVE_FILE_FOLDER, 'models', EXP_NAME + '_epoch-' + str(epoch) + 'ValAcc_{:10.4f}_'.format(epoch_loss) + '.pth.tar')))

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}, ROC:{}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc, epoch_roc))
            stop_time = timeit.default_timer()
            # print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(SAVE_FILE_FOLDER, EXP_NAME + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(SAVE_FILE_FOLDER, EXP_NAME + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0
            # running_roc = 0.0
            list_pred = list()
            list_label = list()

            # for inputs, labels in tqdm(test_dataloader):
            run_count = 0
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # try:
                #     running_roc += roc_auc_score(labels.data.cpu(), preds.cpu())
                # except:
                #     running_roc += 0.5
                # run_count += 1
                list_label += labels.data.cpu().tolist()
                list_pred += preds.cpu().tolist()

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size
            epoch_roc = multiclass_roc_score(label=list_label, pred=list_pred, n_cls=N_CLASSES)

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)
            writer.add_scalar('data/test_roc_epoch', epoch_roc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc:{} ROC: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc, epoch_roc))
            stop_time = timeit.default_timer()
            # print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


if __name__ == "__main__":
    train_model()