import torch
from torch.utils.data import DataLoader

import argparse
import os
from tqdm import tqdm

from data.stanford_cars import CarsDataset
from data.cifar import CustomCIFAR10, CustomCIFAR100, cifar_10_root, cifar_100_root
from data.herbarium_19 import HerbariumDataset19, herbarium_dataroot
from data.augmentations import get_transform
from data.imagenet import get_imagenet_100_gcd_datasets
from data.data_utils import MergedDataset
from data.cub import CustomCub2011, cub_root
from data.fgvc_aircraft import FGVCAircraft, aircraft_root

from utils.general_utils import str2bool
from copy import deepcopy

from utils.config import feature_extract_dir, dino_pretrain_path

### >>>
from models import vpt_vision_transformer as vpt_vit


### <<<


def extract_features_dino(model, loader, save_dir):
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):

            images, labels, idxs = batch[:3]
            images = images.to(device)

            features = model(images)  # CLS_Token for ViT, Average pooled vector for R50

            # Save features
            for f, t, uq in zip(features, labels, idxs):
                t = t.item()
                uq = uq.item()

                save_path = os.path.join(save_dir, f'{t}', f'{uq}.npy')
                torch.save(f.detach().cpu().numpy(), save_path)


def extract_features_timm(model, loader, save_dir):
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):

            images, labels, idxs = batch[:3]
            images = images.to(device)

            features = model.forward_features(images)  # CLS_Token for ViT, Average pooled vector for R50

            # Save features
            for f, t, uq in zip(features, labels, idxs):
                t = t.item()
                uq = uq.item()

                save_path = os.path.join(save_dir, f'{t}', f'{uq}.npy')
                torch.save(f.detach().cpu().numpy(), save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--root_dir', type=str, default=feature_extract_dir)
    parser.add_argument('--warmup_model_dir',
                        type=str,
                        default=None)
    parser.add_argument('--use_best_model', type=str, default='_best')
    parser.add_argument('--model_name', type=str, default='vpt-model', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset', type=str, default='aircraft', help='options: cifar10, cifar100, scars')
    parser.add_argument('--transform', type=str, default='imagenet')

    parser.add_argument('--vpt_dropout', type=float, default=0.0)
    parser.add_argument('--num_prompts', type=int, default=5)
    parser.add_argument('--n_shallow_prompts', type=int, default=0)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--with_parallel', type=str2bool, default=False)
    parser.add_argument('--save_name', type=str, default='')
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device(args.device)

    args.save_dir = os.path.join(args.root_dir, f'{args.dataset}/{args.save_name}')
    print(args)

    print('Loading model...')
    # ----------------------
    # MODEL
    # ----------------------
    if args.model_name == 'vit_dino':

        extract_features_func = extract_features_dino
        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = dino_pretrain_path

        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)

        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

        _, val_transform = get_transform(args.transform, image_size=224, args=args)

    elif args.model_name == 'vpt-model':
        extract_features_func = extract_features_dino
        args.interpolation = 3
        args.crop_pct = 0.875
        model = vpt_vit.__dict__['vit_base'](
            num_prompts=args.num_prompts,
            vpt_dropout=args.vpt_dropout,
            n_shallow_prompts=args.n_shallow_prompts,
        )

        _, val_transform = get_transform(args.transform, image_size=224, args=args)
    else:

        raise NotImplementedError

    if args.warmup_model_dir is not None:

        if args.use_best_model:
            args.warmup_model_dir = args.warmup_model_dir[:-3] + f'{args.use_best_model}.pt'

        print(f'Using weights from {args.warmup_model_dir} ...')
        state_dict = torch.load(args.warmup_model_dir, map_location='cpu')
        if args.with_parallel == True:
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items() if k[:len('module.')] == 'module.'}
            model.load_state_dict(state_dict)
            model = model.to(args.device)
        else:
            model.load_state_dict(state_dict)
            model = model.to(args.device)

        print(f'Saving to {args.save_dir}')

    print('Loading data...')
    # ----------------------
    # DATASET
    # ----------------------
    if args.dataset == 'cifar10':

        train_dataset = CustomCIFAR10(root=cifar_10_root, train=True, transform=val_transform)
        test_dataset = CustomCIFAR10(root=cifar_10_root, train=False, transform=val_transform)
        targets = list(set(train_dataset.targets))

    elif args.dataset in ['cifar100']:

        train_dataset = CustomCIFAR100(root=cifar_100_root, train=True, transform=val_transform)
        test_dataset = CustomCIFAR100(root=cifar_100_root, train=False, transform=val_transform)
        targets = list(set(train_dataset.targets))

    elif args.dataset == 'scars':

        train_dataset = CarsDataset(train=True, transform=val_transform)
        test_dataset = CarsDataset(train=False, transform=val_transform)
        targets = list(set(train_dataset.target))
        targets = [i - 1 for i in targets]  # SCars are labelled 1 - 197. Change to 0 - 196

    elif args.dataset == 'herbarium_19':

        train_dataset = HerbariumDataset19(root=os.path.join(herbarium_dataroot, 'small-train'),
                                           transform=val_transform)

        test_dataset = HerbariumDataset19(root=os.path.join(herbarium_dataroot, 'small-validation'),
                                          transform=val_transform)

        targets = list(set(train_dataset.targets))

    elif args.dataset == 'imagenet_100_gcd':
        datasets = get_imagenet_100_gcd_datasets(train_transform=val_transform, test_transform=val_transform,
                                                 train_classes=range(50),
                                                 prop_train_labels=0.5)

        datasets['train_labelled'].target_transform = None
        datasets['train_unlabelled'].target_transform = None

        train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                      unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

        test_dataset = datasets['test']
        targets = list(set(test_dataset.targets))

    elif args.dataset == 'cub':

        train_dataset = CustomCub2011(root=cub_root, transform=val_transform, train=True)
        test_dataset = CustomCub2011(root=cub_root, transform=val_transform, train=False)
        targets = list(set(train_dataset.data.target.values))
        targets = [i - 1 for i in targets]  # SCars are labelled 1 - 200. Change to 0 - 199

    elif args.dataset == 'aircraft':

        train_dataset = FGVCAircraft(root=aircraft_root, transform=val_transform, split='trainval')
        test_dataset = FGVCAircraft(root=aircraft_root, transform=val_transform, split='test')
        targets = list(set([s[1] for s in train_dataset.samples]))

    else:

        raise NotImplementedError

    # ----------------------
    # DATALOADER
    # ----------------------
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('Creating base directories...')
    # ----------------------
    # INIT SAVE DIRS
    # Create a directory for each class
    # ----------------------
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for fold in ('train', 'test'):

        fold_dir = os.path.join(args.save_dir, fold)
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)

        for t in targets:
            target_dir = os.path.join(fold_dir, f'{t}')
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

    # ----------------------
    # EXTRACT FEATURES
    # ----------------------
    # Extract train features
    train_save_dir = os.path.join(args.save_dir, 'train')
    print('Extracting features from train split...')
    extract_features_func(model=model, loader=train_loader, save_dir=train_save_dir)

    # Extract test features
    test_save_dir = os.path.join(args.save_dir, 'test')
    print('Extracting features from test split...')
    extract_features_func(model=model, loader=test_loader, save_dir=test_save_dir)

    print('Done!')
