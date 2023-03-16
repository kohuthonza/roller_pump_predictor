import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import argparse
from torch.utils.data import DataLoader

from safe_gpu.safe_gpu import GPUOwner

from dataset import WaveDataset
from model_wrapper import RegressionModelWrapper


def parseargs():
    print(' '.join(sys.argv))
    parser = argparse.ArgumentParser()

    # Datasets definition
    #
    parser.add_argument('--trn-wave-directory-path', type=str)
    parser.add_argument('--tst-wave-directory-path', action='append', type=str)
    parser.add_argument('--trn-num-workers', default=8, type=int)
    parser.add_argument('--tst-num-workers', default=8, type=int)
    parser.add_argument('--trn-prefetch-factor', default=1, type=int)
    parser.add_argument('--tst-prefetch-factor', default=1, type=int)
    parser.add_argument('--max-pressure', default=60000, type=float)
    parser.add_argument('--max-speed', default=1.6, type=float)

    # Model definition
    #
    parser.add_argument('-n', '--net', required=True, type=str)

    # Training
    #
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--learning-rate', default=0.0003, type=float)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--dropout-rate', default=0.0, type=float)
    parser.add_argument('--start-iteration', default=0, type=int)
    parser.add_argument('--max-iterations', default=0, type=int)

    # Saving models and reporting during training
    #
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-i', '--in-checkpoint', type=str)
    parser.add_argument('-o', '--out-checkpoint', type=str)
    parser.add_argument('-d', '--checkpoint-dir', default='.', type=str)
    parser.add_argument('--test-step', default=500, type=int)
    parser.add_argument('--save-step', default=500, type=int)
    parser.add_argument('--show-dir', type=str)
    parser.add_argument('--export-dir', type=str)

    parser.add_argument('--view-step', type=int, help='Set test_step, save_step to view_step.')

    args = parser.parse_args()
    return args


def main():
    args = parseargs()

    print("INIT DATASETS")
    print()
    trn_dataset, tst_datasets = init_datasets(trn_wave_directory_path=args.trn_wave_directory_path,
                                              tst_wave_directory_path=args.tst_wave_directory_path,
                                              batch_size=args.batch_size,
                                              max_pressure=args.max_pressure,
                                              max_speed=args.max_speed,
                                              trn_num_workers=args.trn_num_workers,
                                              tst_num_workers=args.tst_num_workers,
                                              trn_prefetch_factor=args.trn_prefetch_factor,
                                              tst_prefetch_factor=args.tst_prefetch_factor)

    gpu_owner = GPUOwner()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print()

    print(f'CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}')

    print()
    print("INIT TRAINING")

    model = RegressionModelWrapper.build_model(net=args.net)
    optimizer = RegressionModelWrapper.build_optimizer(args.optimizer, model, args.learning_rate)

    model_wrapper = RegressionModelWrapper(net=model, optimizer=optimizer, device=device)

    print()
    print("NET")
    print()
    print(model_wrapper.net)

    print()
    load_weights(model_wrapper, args.in_checkpoint, args.start_iteration, args.checkpoint_dir)
    print()

    if args.view_step is not None:
        args.test_step = args.view_step
        args.save_step = args.view_step

    def is_valid_iter(it_no, step):
        return it_no % step == 0

    training_time = 0.0
    total_nb_waves = 0.0

    trn_loss = 0.0
    train_net_time = 0.0
    train_timer = time.time()

    model_wrapper.set_train()
    iteration = args.start_iteration
    stop_training = False

    if iteration >= args.max_iterations and args.test:
        print()
        print(f"TESTING")
        print("---------------------------------------------------------------------------------------------------")
        test_datasets(model_wrapper, trn_dataset, tst_datasets, iteration, args.show_dir, args.export_dir)
        print("---------------------------------------------------------------------------------------------------")
        print()
        return

    if trn_dataset is None:
        return

    while True:
        for batch in trn_dataset:
            test_and_show = args.test and (is_valid_iter(iteration, args.test_step) or iteration == args.start_iteration)
            show_train = is_valid_iter(iteration, args.test_step) and iteration > args.start_iteration
            save_checkpoint = (is_valid_iter(iteration, args.save_step) and iteration > args.start_iteration) or iteration == 0

            if test_and_show or show_train or save_checkpoint:
                print()
                print(f"ITERATION {iteration}")
                print("---------------------------------------------------------------------------------------------------")

            if save_checkpoint:
                save_weights(model_wrapper, args.out_checkpoint, iteration, args.checkpoint_dir)

            if test_and_show:
                test_datasets(model_wrapper, trn_dataset, tst_datasets, iteration, args.show_dir, args.export_dir)
                print()

            if show_train:
                train_time = time.time() - train_timer
                training_time += train_time
                data = batch['input_pressure_wave']
                net_speed = (args.test_step * data.shape[0]) / train_net_time
                trn_loss /= args.test_step
                print(f"TRAIN {iteration} ({total_nb_waves / 1000:.1f}k waves seen) loss:{trn_loss:.3f}"
                      f" lr:{args.learning_rate} time:{train_time:.1f} net_speed:{net_speed}")
                print()
                train_net_time = 0.0
                trn_loss = 0.0
                train_timer = time.time()

            if test_and_show or show_train or save_checkpoint:
                print("---------------------------------------------------------------------------------------------------")
                print()

            if iteration == args.max_iterations:
                stop_training = True
                break

            net_t1 = time.time()

            _, loss = model_wrapper.train_step(batch)

            train_net_time += time.time() - net_t1
            loss = loss.mean()
            trn_loss += loss
            total_nb_waves += len(batch['input_pressure_wave'])

            iteration += 1

        if stop_training:
            break

    training_time += time.time() - train_timer

    print("AVERAGE TIME OF 100 ITERATIONS: {}".format((training_time / (args.max_iterations - args.start_iteration)) * 100))


def test_datasets(model_wrapper, trn_dataset, tst_datasets, iteration, show_dir, export_dir):
    model_wrapper.set_eval()
    if trn_dataset is not None:
        test(model_wrapper, iteration, trn_dataset, train=True, show_dir=show_dir, export_dir=export_dir)
    if tst_datasets:
        for tst_dataset in tst_datasets:
            test(model_wrapper, iteration, tst_dataset, train=False, show_dir=show_dir, export_dir=export_dir)
    model_wrapper.set_train()


def test(model_wrapper, iteration, dataset, train, show_dir, export_dir):
    total_loss = 0
    t1 = time.time()
    total_net_time = 0
    total_nb_waves = 0

    all_targets = []
    all_outputs = []
    all_paths = []

    with torch.no_grad():
        for it_count, batch in enumerate(dataset, 1):
            targets = batch['target_speed_wave']
            paths = batch['path']

            net_t1 = time.time()
            outputs, loss = model_wrapper.test_step(batch)
            total_net_time += time.time() - net_t1
            total_loss += loss.mean().item()
            total_nb_waves += targets.shape[0]
            all_targets.append(targets)
            all_outputs.append(outputs)
            all_paths.append(paths)

    t2 = time.time()

    print('TEST {} {:d} loss:{:.5f} full_speed:{:.0f} net_speed:{:.0f} time:{:.5f}'.format(
        dataset.dataset.wave_directory_path,
        iteration,
        total_loss / it_count,
        total_nb_waves / (t2 - t1),
        total_nb_waves / total_net_time,
        t2 - t1))

    if show_dir is not None:
        show_waves(all_targets, all_outputs, all_paths, dataset.dataset.max_speed, iteration, dataset.dataset.wave_directory_path, train, show_dir)
    if export_dir is not None:
        export_waves(all_outputs, all_paths, dataset.dataset.max_speed, iteration, dataset.dataset.wave_directory_path, export_dir)


def show_waves(all_targets, all_outputs, all_paths, max_speed, iteration, wave_directory_path, train, show_dir):
    iteration_dir = os.path.join(show_dir, os.path.basename(wave_directory_path), str(iteration))
    if not os.path.exists(iteration_dir):
        os.makedirs(iteration_dir)

    counter = 0
    for targets, outputs, paths in zip(all_targets, all_outputs, all_paths):
        for target, output, path in zip(targets, outputs, paths):
            target = target.cpu().numpy()
            output = output.cpu().numpy()
            target += 1
            target = target * (max_speed / 2.0)
            output += 1
            output = output * (max_speed / 2.0)
            t = np.arange(0, 1, 0.001)
            ax = plt.gca()
            ax.set_ylim([0, max_speed])
            plt.plot(t, target, color='red', label='target')
            plt.plot(t, output, color='blue', label='output', linewidth=0.7)
            plt.legend()
            plt.savefig(os.path.join(iteration_dir, '{}.pdf'.format(path)))
            plt.cla()
            if train:
                counter += 1


def export_waves(all_outputs, all_paths, max_speed, iteration, wave_directory_path, export_dir):
    iteration_dir = os.path.join(export_dir, os.path.basename(wave_directory_path), str(iteration))
    if not os.path.exists(iteration_dir):
        os.makedirs(iteration_dir)

    for outputs, paths in zip(all_outputs, all_paths):
        for output, path in zip(outputs, paths):
            output = output.cpu().numpy()
            output += 1
            output = output * (max_speed / 2.0)
            out_lines = []
            with open(os.path.join(wave_directory_path, path)) as f:
                for l, o in zip(f.readlines(), output):
                    out_lines.append("{},{}\n".format(l.strip(), str(o)))
            with open(os.path.join(iteration_dir, path), "w") as f:
                f.writelines(out_lines)


def init_datasets(trn_wave_directory_path, tst_wave_directory_path, batch_size, max_pressure=60000, max_speed=1.6,
                  trn_num_workers=1, tst_num_workers=1, trn_prefetch_factor=100, tst_prefetch_factor=100):

    trn_dataset = None
    if trn_wave_directory_path is not None:
        trn_dataset = WaveDataset(trn_wave_directory_path, max_pressure=max_pressure, max_speed=max_speed)
        trn_dataset = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=trn_num_workers, prefetch_factor=trn_prefetch_factor,
                                 persistent_workers=True)

    tst_datasets = None
    if tst_wave_directory_path is not None:
        tst_datasets = []
        for tst_directory_path in tst_wave_directory_path:
            tst_dataset = WaveDataset(tst_directory_path, max_pressure=max_pressure, max_speed=max_speed)
            tst_dataset = DataLoader(tst_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=tst_num_workers, prefetch_factor=tst_prefetch_factor,
                                     persistent_workers=True)
            tst_datasets.append(tst_dataset)

    return trn_dataset, tst_datasets


def load_weights(training, in_checkpoint=None, start_iteration=0, checkpoint_dir=None):
    checkpoint_path = None
    if in_checkpoint is not None:
        checkpoint_path = in_checkpoint
    elif start_iteration:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_{:06d}.pth".format(start_iteration))
    if checkpoint_path is not None:
        print("LOAD WEIGHTS:", checkpoint_path)
        training.load_weights(checkpoint_path)


def save_weights(training, out_checkpoint, iteration, checkpoint_dir):
    if out_checkpoint is not None:
        checkpoint_path = out_checkpoint
    else:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_{:06d}.pth".format(iteration))
    training.save_weights(checkpoint_path)
    print("CHECKPOINT SAVED TO: {}".format(checkpoint_path))


if __name__ == '__main__':
    main()