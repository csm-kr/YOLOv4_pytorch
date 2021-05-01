import time
import os
import torch
from config import device


def train(epoch, vis, train_loader, model, criterion, optimizer, scheduler, opts):

    print('Training of epoch [{}]'.format(epoch))
    tic = time.time()
    model.train()

    # FIXME warm up --> burn in 1000 batch (iter) 로 바꿔야함
    # if epoch < 5:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = opts.lr * 0.2 * (epoch + 1)
    # elif epoch == 5:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = opts.lr

    for idx, datas in enumerate(train_loader):

        # burn in process
        if opts.burn_in is not None:
            burn_in_idx = idx + epoch * len(train_loader)
            if burn_in_idx < opts.burn_in:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = opts.lr * ((burn_in_idx + 1) / opts.burn_in) ** 4

        images = datas[0]
        boxes = datas[1]
        labels = datas[2]

        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        pred = model(images)
        loss, losses = criterion(pred, boxes, labels)

        # sgd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        toc = time.time()

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # for each steps
        if idx % opts.vis_step == 0 or idx == len(train_loader) - 1:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'Learning rate: {lr:.7f} s \t'
                  'Time : {time:.4f}\t'
                  .format(epoch, idx, len(train_loader),
                          loss=loss,
                          lr=lr,
                          time=toc - tic))

            if vis is not None:
                # loss plot
                vis.line(X=torch.ones((1, 6)).cpu() * idx + epoch * train_loader.__len__(),  # step
                         Y=torch.Tensor([loss, losses[0], losses[1], losses[2], losses[3], losses[4]])
                         .unsqueeze(0).cpu(),
                         win='train_loss',
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='Loss',
                                   title='training loss',
                                   legend=['Total Loss', 'xy Loss', 'wh Loss', 'obj Loss', 'no obj Loss', 'cls Loss']))

    # # 각 epoch 마다 저장
    if not os.path.exists(opts.save_path):
        os.mkdir(opts.save_path)

    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict()}

    torch.save(checkpoint, os.path.join(opts.save_path, opts.save_file_name + '.{}.pth.tar'.format(epoch)))


