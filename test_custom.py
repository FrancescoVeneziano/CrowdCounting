import torch
import os
import numpy as np
from datasets.crowd_custom import Crowd
from models.vgg import vgg19
import argparse
from matplotlib import cm as c
import matplotlib.pyplot as plt

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='/home/teddy/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/home/teddy/vgg',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(args.data_dir, 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))

    frames_num = []
    predictions = []
    for inputs, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            frames_num.append(int(name[0]))
            predictions.append(torch.sum(outputs).item())
            print(f'Frame: {name}, Predicted count: {torch.sum(outputs).item()}')
        temp = np.asarray(outputs.detach().cpu().reshape(outputs.detach().cpu().shape[2],outputs.detach().cpu().shape[3]))
        fig, ax = plt.subplots(nrows=2, ncols=1)
        fig.tight_layout()
        ax[0].set_title('Original image')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].imshow(plt.imread('/content/Bayesian-Crowd-Counting/datasets/frames/' + name[0] + '.jpg'))
        ax[1].set_title(f'Predicted image: {round(torch.sum(outputs).item())} people')
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].imshow(temp, cmap=c.jet)
        plt.savefig('/content/plot_predictions/pred_' + name[0] + '.jpg')
    figure = plt.figure(figsize=(12, 8))
    plt.title('Count of people over frames')
    plt.xlabel('Frame number')
    plt.ylabel('Number of people')
    plt.grid()
    plt.plot(frames_num, predictions, 'r-o')
    plt.savefig('/content/plot_predictions/plot_preds-frames.jpg')
