# Common libs
import time
import os
import numpy as np

# My libs
from utils.config import Config
from utils.tester import ModelTester

# Use the model that matches the RoITr-Lite KITTI checkpoint
from models.KPFCNN_model_kitti_roitr_lite import KernelPointFCNN

# Datasets
from datasets.ETH import ETHDataset


def find_latest_roitr_lite_log():
    logs = np.sort([
        os.path.join('results_kitti', f)
        for f in os.listdir('results_kitti')
        if f.startswith('D3Feat_KITTI_ROITR_LITE_')
    ])

    if len(logs) == 0:
        raise ValueError(
            'No RoITr-Lite KITTI log found under '
            'results_kitti/D3Feat_KITTI_ROITR_LITE_*'
        )

    return logs[-1]


def test_caller(path, step_ind=-1, on_val=True):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    # Load model parameters from the KITTI-trained RoITr-Lite checkpoint
    config = Config()
    config.load(path)

    # Adapt KITTI-trained model settings to ETH testing
    config.first_subsampling_dl = 0.5
    config.dataset = 'ETH'
    config.KP_extent = 2
    config.batch_num = 1
    config.saving = False

    print()
    print('Dataset Preparation')
    print('*******************')

    dataset = ETHDataset(1, load_test=True)
    dataset.init_test_input_pipeline(config)

    print('Creating Model')
    print('**************\n')
    t1 = time.time()

    model = KernelPointFCNN(dataset.flat_inputs, config)

    # Find snapshot
    snap_path = os.path.join(path, 'snapshots')
    snap_steps = [
        int(f[:-5].split('-')[-1])
        for f in os.listdir(snap_path)
        if f.endswith('.meta')
    ]

    if len(snap_steps) == 0:
        raise ValueError('No snapshot meta file found in: ' + snap_path)

    chosen_step = np.sort(snap_steps)[step_ind]
    chosen_snap = os.path.join(path, 'snapshots', 'snap-{:d}'.format(chosen_step))

    print('Chosen log:', path)
    print('Chosen snapshot:', chosen_snap)

    tester = ModelTester(model, restore_snap=chosen_snap)

    t2 = time.time()
    print('\n----------------')
    print('Done in {:.1f} s'.format(t2 - t1))
    print('----------------\n')

    print('Start Test')
    print('**********\n')

    tester.generate_descriptor(model, dataset)


if __name__ == '__main__':

    # Set a fixed path if needed, for example:
    # chosen_log = 'results_kitti/D3Feat_KITTI_ROITR_LITE_20260607_115020'

    # Default: automatically use the latest RoITr-Lite KITTI-trained model
    chosen_log = None

    chosen_snapshot = -1
    on_val = True

    if chosen_log is None:
        chosen_log = find_latest_roitr_lite_log()

    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exist: ' + chosen_log)

    test_caller(chosen_log, chosen_snapshot, on_val)
