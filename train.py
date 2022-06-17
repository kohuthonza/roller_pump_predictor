import torch
import cv2
import numpy as np
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
    parser.add_argument('--trn-wave-directory-path', required=True, type=str)
    parser.add_argument('--tst-wave-directory-path', action='append', type=str)
    parser.add_argument('--trn-num-workers', default=8, type=int)
    parser.add_argument('--tst-num-workers', default=8, type=int)
    parser.add_argument('--trn-prefetch-factor', default=1, type=int)
    parser.add_argument('--tst-prefetch-factor', default=1, type=int)

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
    parser.add_argument('--max-iterations', default=500000, type=int)

    # Saving models and reporting during training
    #
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-i', '--in-checkpoint', type=str)
    parser.add_argument('-o', '--out-checkpoint', type=str)
    parser.add_argument('-d', '--checkpoint-dir', default='.', type=str)
    parser.add_argument('--save-step', default=500, type=int)
    parser.add_argument('--show-dir', default='.', type=str)

    parser.add_argument('--view-step', type=int, help='Set test_step, save_step to view_step.')

    args = parser.parse_args()
    return args


def main():
    args = parseargs()

    print("INIT DATASETS")
    print()
    trn_dataset, tst_datasets = init_datasets(args.trn_wave_directory_path,
                                              args.tst_wave_directory_path,
                                              args.batch_size,
                                              args.trn_num_workers,
                                              args.tst_num_workers,
                                              args.trn_prefetch_factor,
                                              args.tst_prefetch_factor)
    init_show_dirs(args.show_dir)

    gpu_owner = GPUOwner(args.n_gpu)
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
                model_wrapper.set_eval()
                test(model_wrapper, iteration, trn_dataset, train=True, show_dir=args.show_dir)
                if tst_datasets:
                    for tst_dataset in tst_datasets:
                        test(model_wrapper, iteration, tst_dataset, train=False, show_dir=args.show_dir)
                model_wrapper.set_train()
                print()

            if show_train:
                train_time = time.time() - train_timer
                training_time += train_time
                data = batch['input_pressure_wave']
                net_speed = (args.test_step * data.shape[0]) / train_net_time
                trn_loss /= args.test_step
                print(f"TRAIN {iteration} ({total_nb_waves / 1000:.1f}k lines seen) loss:{trn_loss:.3f} \
                           lr:{args.learning_rate} time:{train_time:.1f} net_speed:{net_speed}")
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

    training_time += time.time() - train_time

    print("AVERAGE TIME OF 100 ITERATIONS: {}".format((training_time / (args.max_iterations - args.start_iteration)) * 100))


def test(model_wrapper, iteration, dataset, train, show_dir):
    total_loss = 0
    t1 = time.time()
    total_net_time = 0
    total_nb_waves = 0

    all_inputs = []
    all_outputs = []

    with torch.no_grad():
        for it_count, batch in enumerate(dataset, 1):
            inputs = batch['input_pressure_wave']

            net_t1 = time.time()
            outputs, loss = model_wrapper.test_step(batch)

            all_inputs.append(inputs)
            all_outputs.append(outputs)

            total_loss += loss.mean().item()
            total_net_time += time.time() - net_t1
            total_nb_waves += inputs.shape[0]

            if train:
                if it_count > 1500 // dataset.batch_size:
                    break

    t2 = time.time()

    print('TEST {} {:d} loss:{:.5f} full_speed:{:.0f} net_speed:{:.0f} time:{:.1f}'.format(
        dataset.wave_directory_path,
        iteration,
        total_loss / it_count,
        total_nb_waves / (t2 - t1),
        total_nb_waves / total_net_time,
        t2 - t1))

    if show_dir is not None:
        show_images(all_inputs, all_outputs, iteration, dataset.wave_directory_path, train, show_dir)


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


def show_images(all_inputs, all_outputs, iteration, wave_directory_path, train, show_dir):
    """
    images = []
    data = data.detach().cpu().numpy()
    data = np.transpose(data, (0, 2, 3, 1))
    data *= 255
    for i in range(data.shape[0]):
        images.append(data[i])

    image = np.concatenate(images, axis=0)

    dataset_name = os.path.split(dataset_name)[-1]
    train_test = "train"
    if not train:
        train_test = "test"

    image_path = os.path.join(show_dir, "test", train_test, "TEST_BATCH_{}_{:06d}.jpg".format(dataset_name, iteration))

    if not train:
        print("SAVING TEST BATCH TO: {}".format(image_path))
    else:
        print("SAVING TRAIN BATCH TO: {}".format(image_path))
    cv2.imwrite(image_path, image)
    """
    pass


def init_show_dirs(show_dir):
    trn_dir_path = os.path.join(show_dir, "train")
    if not os.path.exists(trn_dir_path):
        os.makedirs(trn_dir_path)
    tst_dir_path = os.path.join(show_dir, "test")
    if not os.path.exists(tst_dir_path):
        os.makedirs(tst_dir_path)


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