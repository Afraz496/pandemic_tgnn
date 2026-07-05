import os
import torch
import numpy as np
from math import ceil

import torch.nn.functional as F
import torch.optim as optim

from preprocess import  AverageMeter, generate_batches, generate_batches_lstm

from models import MPNN_LSTM, LSTM, MPNN
import time


def train(model: torch.nn.Module ,
          optimizer: torch.optim.Optimizer  ,
          adj: torch.nn.Module  ,
          features: torch.nn.Module ,
          y: torch.nn.Module ):
    """
    Train the model

    Parameters:
    model (torch.nn.Module): Model to train
    adj (torch.Tensor): Adjacency matrix
    features (torch.Tensor): Features matrix
    y (torch.Tensor): Labels matrix


    Returns:
    output (torch.Tensor): Output predictions of the model
    loss_train (torch.Tensor): Loss of the model
    """
    optimizer.zero_grad()
    output = model(adj, features)
    loss_train = F.mse_loss(output, y)
    loss_train.backward()
    optimizer.step()
    return output, loss_train



def test(model: torch.nn.Module,
         adj: torch.Tensor,
         features: torch.Tensor,
         y: torch.Tensor):
    """
    Test the model

    Parameters:
    model (torch.nn.Module): Model to test
    adj (torch.Tensor): Adjacency matrix
    features (torch.Tensor): Features matrix
    y (torch.Tensor): Labels matrix

    Returns:
    output (torch.Tensor): Output predictions of the model
    loss_test (torch.Tensor): Loss of the model
    """
    with torch.no_grad():
        output = model(adj, features)
        loss_test = F.mse_loss(output, y)
    return output, loss_test



def save_best_checkpoint(model_dir: str, tag: str, state: dict, val_loss: float):
    """
    Persist `state` to <model_dir>/<tag>.pth.tar, but only if its validation
    loss beats the checkpoint already stored there (if any).

    run_neural_model is called once per (model, shift, test_sample), so this
    keeps exactly one file per (model, shift): the fit with the best val loss.

    Returns:
    saved (bool): True if the file was (over)written.
    """
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{tag}.pth.tar")

    if os.path.exists(path):
        try:
            prev = torch.load(path, map_location="cpu")
            if prev.get("val_loss", float("inf")) <= val_loss:
                return False  # existing checkpoint is at least as good
        except Exception:
            pass  # corrupt/old-format file -> overwrite it

    torch.save(state, path)
    return True



def run_neural_model(model: str,
                     n_nodes: int,
                     early_stop: int,
                     idx_train: list,
                     window:  int,
                     shift: int,
                     batch_size: int,
                     y: list,
                     device: torch.device,
                     test_sample: int,
                     graph_window: int,
                     recur: bool,
                     gs_adj: list,
                     features: list,
                     idx_val: list,
                     hidden,dropout: float,
                     lr: float,
                     nfeat: int,
                     epochs: int,
                     print_epoch: int=50,
                     model_dir: str=None,
                     tag: str=None):
    """
    Derive batches from the data, trian the mdoel and test it

    Parameters:
    model (str): Model to use
    n_nodes (int): Number of nodes in the country mobility graph
    early_stop (int): Number of epochs to wait before stopping the training because the validation is not improving
    idx_train (list): List of the training samples
    window (int): Window size
    shift (int): Shift size
    batch_size (int): Batch size
    y (list): Labels
    device (torch.device): Device to use
    test_sample (int): Test sample
    graph_window (int): Window size for the graph
    recur (bool): Whether to use recurrent layers
    gs_adj (list): List of adjacency matrices
    features (list): List of features
    idx_val (list): List of validation samples
    hidden (int): Hidden size
    dropout (float): Dropout rate
    lr (float): Learning rate
    nfeat (int): Number of features
    epochs (int): Number of epochs
    print_epoch (int): Number of epochs to wait before printing the results
    model_dir (str): If set, persist the best checkpoint per (model, shift)
                     to this directory (only kept if it beats the existing one)
    tag (str): Filename stem for the persisted checkpoint,
               e.g. "England_MPNN_LSTM_shift3"

    Returns:
    error (float): Average error per region
    """

    #-------------------- Generate the batches
    if(model=="LSTM"):
        lstm_features = 1*n_nodes
        adj_train, features_train, y_train = generate_batches_lstm(n_nodes, y, idx_train, window, shift,  batch_size,device,test_sample)
        adj_val, features_val, y_val = generate_batches_lstm(n_nodes, y, idx_train, window, shift, batch_size,device,test_sample)
        adj_test, features_test, y_test = generate_batches_lstm(n_nodes, y, [test_sample],  window, shift,  batch_size,device,test_sample)


    elif(model=="MPNN_LSTM"):
        adj_train, features_train, y_train = generate_batches(gs_adj, features, y, idx_train, graph_window, shift, batch_size,device,test_sample)
        adj_val, features_val, y_val = generate_batches(gs_adj, features, y, idx_val, graph_window,  shift,batch_size, device,test_sample)
        adj_test, features_test, y_test = generate_batches(gs_adj, features, y,  [test_sample], graph_window,shift, batch_size, device,test_sample)

    else:
        adj_train, features_train, y_train = generate_batches(gs_adj, features, y, idx_train, 1,  shift,batch_size,device,test_sample)
        adj_val, features_val, y_val = generate_batches(gs_adj, features, y, idx_val, 1,  shift,batch_size,device,test_sample)
        adj_test, features_test, y_test = generate_batches(gs_adj, features, y,  [test_sample], 1,  shift,batch_size, device,-1)


    #-------------------- Training
    model_name = model
    n_train_batches = ceil(len(idx_train) / batch_size)

    if model_name == "LSTM":
        model = LSTM(
            nfeat=lstm_features,
            nhid=hidden,
            n_nodes=n_nodes,
            window=window,
            batch_size=batch_size,
            recur=recur,
        ).to(device)

    elif model_name == "MPNN_LSTM":
        model = MPNN_LSTM(
            nfeat=nfeat,
            nhid=hidden,
            nout=1,
            n_nodes=n_nodes,
            window=graph_window,
            dropout=dropout,
        ).to(device)

    elif model_name == "MPNN":
        model = MPNN(nfeat=nfeat, nhid=hidden, nout=1, dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    # Per-run scratch checkpoint. Tagged so concurrent/interleaved runs of
    # different models or shifts can't clobber each other's file. Written to
    # model_dir when available (guaranteed writable), else the cwd.
    ckpt_name = f"model_best_{tag}.pth.tar" if tag else "model_best.pth.tar"
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        ckpt_tmp = os.path.join(model_dir, ckpt_name)
    else:
        ckpt_tmp = ckpt_name

    #------------------- Train
    best_val_acc = 1e8
    val_among_epochs = []
    train_among_epochs = []

    for epoch in range(epochs):
        start = time.time()

        model.train()
        train_loss = AverageMeter()

        # Train for one epoch
        for batch in range(n_train_batches):
            output, loss = train(model, optimizer, adj_train[batch], features_train[batch], y_train[batch])
            train_loss.update(loss.data.item(), output.size(0))

        # Evaluate on validation set
        model.eval()

        output, val_loss = test(model, adj_val[0], features_val[0], y_val[0])
        val_loss = float(val_loss.detach().cpu().numpy())

        # Print results
        if(epoch%print_epoch==0):
            print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg),"val_loss=", "{:.5f}".format(val_loss), "time=", "{:.5f}".format(time.time() - start))

        train_among_epochs.append(train_loss.avg)
        val_among_epochs.append(val_loss)

        # check if the model is stuck
        if(epoch<30 and epoch>10):
            if(len(set([round(val_e) for val_e in val_among_epochs[-20:]])) == 1 ):
                print("Break becuase it s stuck")
                break

        if( epoch>early_stop):
            if(len(set([round(val_e) for val_e in val_among_epochs[-int(early_stop/2):]])) == 1):
                print("Break early stop")
                break

        #--------- Remember best accuracy and save checkpoint
        if val_loss < best_val_acc:
            best_val_acc = val_loss
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, ckpt_tmp)

        scheduler.step(val_loss)


    #---------------- Testing
    checkpoint = torch.load(ckpt_tmp)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()

    output, loss = test(model,adj_test[0], features_test[0], y_test[0])

    if(model_name=="LSTM"):
        o = output.view(-1).cpu().detach().numpy()
        l = y_test[0].view(-1).cpu().numpy()
    else:
        o = output.cpu().detach().numpy()
        l = y_test[0].cpu().numpy()

    #--------------- Average error per region
    error = np.sum(abs(o-l))/n_nodes

    #---------------- Persist the best checkpoint per (model, shift)
    if model_dir and tag:
        save_best_checkpoint(model_dir, tag, {
            'model_name': model_name,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'val_loss'  : best_val_acc,
            'test_error': float(error),
            'test_sample': test_sample,
            # Everything needed to rebuild the model for inference:
            'hparams': {
                'nfeat': nfeat,
                'hidden': hidden,
                'dropout': dropout,
                'window': window,
                'graph_window': graph_window,
                'n_nodes': n_nodes,
                'batch_size': batch_size,
                'recur': recur,
                'shift': shift,
            },
        }, best_val_acc)

    # Clean up the scratch file so the code dir doesn't fill with temp checkpoints.
    if os.path.exists(ckpt_tmp):
        os.remove(ckpt_tmp)

    return error


def load_checkpoint(path: str, device: torch.device = torch.device("cpu")):
    """
    Rebuild a saved model from a checkpoint written by save_best_checkpoint.

    Returns:
    model (torch.nn.Module): Model in eval mode with weights loaded
    checkpoint (dict): Full checkpoint (hparams, val_loss, etc.)
    """
    ckpt = torch.load(path, map_location=device)
    hp = ckpt['hparams']
    name = ckpt['model_name']

    if name == "LSTM":
        model = LSTM(nfeat=hp['n_nodes'], nhid=hp['hidden'], n_nodes=hp['n_nodes'],
                     window=hp['window'], batch_size=hp['batch_size'], recur=hp['recur'])
    elif name == "MPNN_LSTM":
        model = MPNN_LSTM(nfeat=hp['nfeat'], nhid=hp['hidden'], nout=1,
                          n_nodes=hp['n_nodes'], window=hp['graph_window'],
                          dropout=hp['dropout'])
    else:
        model = MPNN(nfeat=hp['nfeat'], nhid=hp['hidden'], nout=1, dropout=hp['dropout'])

    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model, ckpt