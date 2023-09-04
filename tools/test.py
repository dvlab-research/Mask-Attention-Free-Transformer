import argparse
import gorilla
import torch
from tqdm import tqdm
import os

from maft.dataset import build_dataloader, build_dataset
from maft.evaluation import ScanNetEval
from maft.utils import get_root_logger, save_gt_instances, save_pred_instances
from tools.train import get_model

def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--out', type=str, help='directory for output results')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cfg = gorilla.Config.fromfile(args.config)
    gorilla.set_random_seed(cfg.test.seed)
    logger = get_root_logger()

    # model = SPFormer(**cfg.model).cuda()
    model_name = cfg.model.pop("name", "SPFormer")
    model = get_model(cfg, model_name)
    cfg.model_name = model_name

    if not os.path.exists(os.path.join(cfg.data.train.data_root, "train")):
        if os.path.exists("/mnt/proj78/ykchen/dataset/scannetv2_spformer/train/scene0000_00_inst_nostuff.pth"):
            cfg.data.train.data_root = "/mnt/proj78/ykchen/dataset/scannetv2_spformer"
            cfg.data.val.data_root = "/mnt/proj78/ykchen/dataset/scannetv2_spformer"
            cfg.data.test.data_root = "/mnt/proj78/ykchen/dataset/scannetv2_spformer"
        elif os.path.exists("/mnt/proj74/xinlai/inst_seg/SPFormer/data/scannetv2/train/scene0000_00_inst_nostuff.pth"):
            cfg.data.train.data_root = "/mnt/proj74/xinlai/inst_seg/SPFormer/data/scannetv2"
            cfg.data.val.data_root = "/mnt/proj74/xinlai/inst_seg/SPFormer/data/scannetv2"
            cfg.data.test.data_root = "/mnt/proj74/xinlai/inst_seg/SPFormer/data/scannetv2"
        elif os.path.exists("/dataset/xinlai/dataset/scannet_spformer/train/scene0000_00_inst_nostuff.pth"):
            cfg.data.train.data_root = "/dataset/xinlai/dataset/scannet_spformer"
            cfg.data.val.data_root = "/dataset/xinlai/dataset/scannet_spformer"
            cfg.data.test.data_root = "/dataset/xinlai/dataset/scannet_spformer"

    logger.info(f'Load state dict from {args.checkpoint}')
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)

    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.test)

    results, scan_ids, pred_insts, gt_insts = [], [], [], []

    progress_bar = tqdm(total=len(dataloader))
    with torch.no_grad():
        model.eval()
        for batch in dataloader:

            if cfg.train.get("use_rgb", True) == False:
                batch['feats'] = batch['feats'][:, 3:]

            if cfg.model_name.startswith("SPFormer"):
                batch.pop("coords_float", "")

            if not cfg.model_name.endswith("no_superpoint"):
                batch.pop("batch_points_offsets", "")

            result = model(batch, mode='predict')
            results.append(result)
            progress_bar.update()
        progress_bar.close()

    for res in results:
        scan_ids.append(res['scan_id'])
        pred_insts.append(res['pred_instances'])
        gt_insts.append(res['gt_instances'])

    if not cfg.data.test.prefix == 'test':
        logger.info('Evaluate instance segmentation')
        scannet_eval = ScanNetEval(dataset.CLASSES)
        scannet_eval.evaluate(pred_insts, gt_insts)

    # save output
    if args.out:
        logger.info('Save results')
        nyu_id = dataset.NYU_ID
        save_pred_instances(args.out, 'pred_instance', scan_ids, pred_insts, nyu_id)
        if not cfg.data.test.prefix == 'test':
            save_gt_instances(args.out, 'gt_instance', scan_ids, gt_insts, nyu_id)


if __name__ == '__main__':
    main()
