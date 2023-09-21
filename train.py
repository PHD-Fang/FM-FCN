import os
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils import weights_init, get_lr_scheduler, set_optimizer_lr, get_lr
from utils.loss import loss_fn
from datasets.rppg_dataset import RppgDataset
from utils.get_hr import EvaluateHR
import time
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CUDA = torch.cuda.is_available()

if __name__ == '__main__':
    config_file = r"config/TRAIN_PURE_FMFCN.yaml"

    # ---------------------------------------------------------------------#
    #   config
    # ---------------------------------------------------------------------#
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    # ---------------------------------------------------------------------#
    #   init
    # ---------------------------------------------------------------------#
    log_dir = config["WORK_PATH"]
    work_name = time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()) + '_' + config['MODEL']['NAME']
    save_dir = log_dir + '/' + work_name + "/"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(save_dir + "log.txt", 'w+') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    # ---------------------------------------------------------------------#
    #   build model
    # ---------------------------------------------------------------------#
    model_name = config['MODEL']['NAME']
    if model_name == 'FM_FCN':
        from nets.FM_FCN import FM_FCN
        model = FM_FCN(in_channels=3)
    else:
        raise ValueError(f'不支持{model_name}模型！')
    weights_init(model)
    if CUDA:
        model = model.cuda()

    # ---------------------------------------------------------------------#
    #    build dataset
    # ---------------------------------------------------------------------#
    train_dataset = RppgDataset(config['TRAIN']['DATA']['DATA_PATH'],
                                config['TRAIN']['DATA']['PREPROCESS']['IMG_SIZE'],
                                config['TRAIN']['DATA']['PREPROCESS']['TIME_LENGTH'],
                                config['TRAIN']['DATA']['PREPROCESS']['OVERLAP'],
                                config['TRAIN']['DATA']['ST'],
                                config['TRAIN']['DATA']['ED'],
                                True,
                                config['TRAIN']['DATA']['PREPROCESS']['MODE'],
                                diff_norm=config['TRAIN']['DATA']['PREPROCESS']['DIFF_NORM_METHOD'],
                                video_norm_per_channel=config['TRAIN']['DATA']['PREPROCESS']['VIDEO_NORM_PER_CHANNEL'])
    val_dataset = RppgDataset(config['VAL']['DATA']['DATA_PATH'],
                              config['VAL']['DATA']['PREPROCESS']['IMG_SIZE'],
                              config['VAL']['DATA']['PREPROCESS']['TIME_LENGTH'],
                              config['VAL']['DATA']['PREPROCESS']['OVERLAP'],
                              config['VAL']['DATA']['ST'],
                              config['VAL']['DATA']['ED'],
                              False,
                              config['VAL']['DATA']['PREPROCESS']['MODE'],
                              diff_norm=config['VAL']['DATA']['PREPROCESS']['DIFF_NORM_METHOD'],
                              video_norm_per_channel=config['VAL']['DATA']['PREPROCESS']['VIDEO_NORM_PER_CHANNEL'])
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config['TRAIN']['BATCH_SIZE'],
                              num_workers=config['DEVICE']['NUM_WORKERS'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config['VAL']['BATCH_SIZE'],
                            num_workers=config['DEVICE']['NUM_WORKERS'], pin_memory=True, drop_last=True)

    # ---------------------------------------------------------------------#
    #   build loss
    # ---------------------------------------------------------------------#
    loss_func = loss_fn(config['TRAIN']['LOSS_TYPE'])

    # ---------------------------------------------------------------------#
    #   train and val steps
    # ---------------------------------------------------------------------#
    train_epoch_step = len(train_loader)
    val_epoch_step = len(val_loader)

    # ---------------------------------------#
    #   select optimizer
    # ---------------------------------------#
    init_lr = config['TRAIN']['INIT_LR']
    optim_name = config['TRAIN']['OPTIMIZER_TYPE']
    if optim_name == 'adam':
        optimizer = optim.Adam(model.parameters(), init_lr)
    elif optim_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), init_lr)
    elif optim_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), init_lr, momentum=0.937, nesterov=True)
    else:
        raise ValueError(f'不支持{optim_name}优化器！')

    # ---------------------------------------#
    #   select lr decay type
    # ---------------------------------------#
    min_lr = config['TRAIN']['MIN_LR']
    lr_decay_type = config['TRAIN']['LR_DECAY_TYPE']
    epochs = config['TRAIN']['EPOCHS']
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr, min_lr, epochs)

    # ---------------------------------------------------------------------#
    #   heart rate evaluator
    # ---------------------------------------------------------------------#
    EvalHR = EvaluateHR(mode=config['TRAIN']['DATA']['PREPROCESS']['MODE'])

    for epoch in range(epochs):
        # ---------------------------------------------------------------------#
        #    modify lr
        # ---------------------------------------------------------------------#
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        # ---------------------------------------------------------------------#
        #   start training
        # ---------------------------------------------------------------------#
        train_loss = 0
        model.train()
        with tqdm(total=train_epoch_step, desc=f'[Train] {model_name} Epoch {epoch + 1}/{epochs}', ncols=100, postfix=dict,
                  mininterval=0.3) as pbar:
            for iteration, batch in enumerate(train_loader):
                videos, labels = batch[0].squeeze(), batch[1].squeeze()
                if CUDA:
                    with torch.no_grad():
                        if isinstance(videos, list):
                            for i in range(len(videos)):
                                videos[i] = videos[i].cuda()
                        else:
                            videos = videos.cuda()
                        labels = labels.cuda()

                optimizer.zero_grad()
                outputs = model(videos).squeeze()
                loss_value = loss_func(outputs, labels)
                loss_value.backward()
                train_loss += loss_value.item()
                pbar.set_postfix(**{'loss': train_loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                optimizer.step()
                pbar.update(1)

        # ---------------------------------------------------------------------#
        #   start val
        # ---------------------------------------------------------------------#
        model.eval()
        EvalHR.clear()
        val_loss = 0
        with tqdm(total=val_epoch_step, desc=f'[Valid] {model_name} Epoch {epoch + 1}/{epochs}', ncols=100, postfix=dict,
                  mininterval=0.3) as pbar:
            for iteration, batch in enumerate(val_loader):
                videos, labels = batch[0].squeeze(), batch[1].squeeze()
                if CUDA:
                    with torch.no_grad():
                        if isinstance(videos, list):
                            for i in range(len(videos)):
                                videos[i] = videos[i].cuda()
                        else:
                            videos = videos.cuda()
                        labels = labels.cuda()
                        
                outputs = model(videos).squeeze()
                loss_value = loss_func(outputs, labels)
                val_loss += loss_value.item()
                # ---------------------------------------------------------------------#
                #   cal heart rate
                # ---------------------------------------------------------------------#
                outputs = outputs.detach().cpu().numpy().ravel()
                labels = labels.detach().cpu().numpy().ravel()
                if config['VAL']['BATCH_SIZE'] > 1:
                    for i in range(len(labels)):
                        EvalHR.add_data(outputs[i], labels[i])
                else:
                    EvalHR.add_data(outputs, labels)

                pbar.set_postfix(**{'loss': val_loss / (iteration + 1)})
                pbar.update(1)
        print(f'Total Loss: {train_loss / train_epoch_step:.4f} || Val Loss: {val_loss / val_epoch_step:.4f} ')

        # ---------------------------------------------------------------------#
        #   metrics
        # ---------------------------------------------------------------------#
        hr_loss = EvalHR.get_loss()
        hr_str = ""
        for key in hr_loss:
            print(f"\t{key}: {hr_loss[key]:.4f}")
            hr_str += f"-{key}[{hr_loss[key]:.4f}]"
        with open(save_dir + "log.txt", 'a+') as f:
            f.write(f"Epoch {epoch + 1}, ")
            f.write(f"loss: {train_loss / train_epoch_step:.4f}, val_loss: {val_loss / val_epoch_step:.4f}")
            for key in hr_loss:
                f.write(f", {key}: {hr_loss[key]:.4f}")
            f.write("\n")

        # ---------------------------------------------------------------------#
        #   save
        # ---------------------------------------------------------------------#
        save_state_dict = model.state_dict()
        torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss[%.4f]-val_loss[%.4f]%s.pth" % (
                epoch + 1, train_loss / train_epoch_step, val_loss / val_epoch_step, hr_str)))
