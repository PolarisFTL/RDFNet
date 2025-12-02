import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import get_lr

def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, gen, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    Dehazy_loss = 0
    loss_detection = 0
    criterion = nn.MSELoss()

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
        model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets, clean = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                clean = clean.cuda(local_rank)
                hazy_and_clear = torch.cat([images, clean], dim = 0).cuda()
        optimizer.zero_grad()

        if not fp16:
            outputs         = model_train(hazy_and_clear)
            detect_outputs = [outputs[0],outputs[1],outputs[2]]
            loss_detection      = yolo_loss(detect_outputs, targets, images)
            loss_dehazy     = criterion(outputs[3], clean)
            loss_value      = 1 * loss_detection + 0.1 * loss_dehazy
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs         = model_train(images)
                loss_value      = yolo_loss(outputs, targets, images)
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)
        Dehazy_loss += loss_dehazy.item()
        loss += loss_value.item()
        loss_detection = (loss - 0.1 * Dehazy_loss)
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1),
                                'loss_detection'  : loss_detection / (iteration + 1),
                                'Dehazy_loss': Dehazy_loss / (iteration + 1),
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, loss / epoch_step)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (loss / epoch_step))
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f.pth" % (epoch + 1, loss / epoch_step)))
        if loss / epoch_step <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
