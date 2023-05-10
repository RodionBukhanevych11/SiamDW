import torch

class Config:
    images_root = '/home/rodion/SiamDW/data/archive_1/images'
    annots_train_path = '/home/rodion/SiamDW/data/archive_1/train.json'
    annots_val_path = '/home/rodion/SiamDW/data/archive_1/val.json'
    save_log_dir = 'logs/'
    seed = 42
    save_best = True
    metrics_file = 'metrics.txt'
    lr = 0.0005
    min_lr = 1e-6
    t_max = 20
    num_epochs = 30
    batch_size = 5
    simple_combs = False
    img_size = {'height': 125, 'width': 125}
    accum = 1
    precision = 32
    triplets = False
    num_workers = 4
    neptune_run_object = None
    neptune_project = "platezhkina13/SiameseDronesCLS"
    neptune_api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MWU0OTI0ZC00MjlkLTRmYjktYTc5Yi0yOGUzZjVjZGQzZGUifQ=="
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
