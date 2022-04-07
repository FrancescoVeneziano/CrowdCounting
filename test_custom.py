import torch
import os
import cv2
from glob import glob
import numpy as np
from datasets.crowd_custom import Crowd
from models.vgg import vgg19
import argparse
from PIL import Image

args = None

def read_video(video):
    FRAMES_DIR = '/content/Bayesian-Crowd-Counting/datasets/frames/'
    SKIP = 5
    vidcap = cv2.VideoCapture(video)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    success, image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(f'{FRAMES_DIR}{str(count).zfill(5)}.jpg', image)     # save frame as JPEG file      
      success, image = vidcap.read()
      count += SKIP
      vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
    return FRAMES_DIR, width, height

def write_count(img, text):
    TEXT_COLOR = (0, 0, 0)
    BOX_COLOR = (255, 255, 255)
    TEXT_SIZE = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'{text} people', (10, 50), font, TEXT_SIZE, BOX_COLOR, 16, cv2.LINE_AA)
    cv2.putText(img, f'{text} people', (10, 50), font, TEXT_SIZE, TEXT_COLOR, 4, cv2.LINE_AA)
    return img


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--video', default='/content/drive/Shareddrives/InnLab_Carbonaro/CrowdCounting/drone_stock/brazil_like.mp4',
                        help='model directory')
    parser.add_argument('--save-dir', default='/home/teddy/vgg',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    data_dir, width, height = read_video(args.video)
    FPS_WRITE = 6
    vid_writer = cv2.VideoWriter(
        '/content/plot_predictions/final_video.mp4', cv2.VideoWriter_fourcc(*"mp4v"), FPS_WRITE, (int(width), int(height))
    )
    datasets = Crowd(data_dir, 512, 8, is_gray=False, method='val')
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
            predictions.append(round(torch.sum(outputs).item()))
            #print(f'Frame: {name}, Predicted count: {torch.sum(outputs).item()}')
    for i, frame_path in enumerate(sorted(glob(os.path.join(data_dir, '*.jpg')))):
        frame = cv2.imread(frame_path)
        frame_output = write_count(frame, predictions[i])
        vid_writer.write(frame_output)
        '''
        temp = np.asarray(outputs.detach().cpu().reshape(outputs.detach().cpu().shape[2],outputs.detach().cpu().shape[3]))
        fig, ax = plt.subplots()
        fig.tight_layout()
        #ax.set_title('Original image')
        ax.set_xticks([])
        ax.set_yticks([])
        at = AnchoredText(f'{round(torch.sum(outputs).item())} people', prop=dict(size=10), frameon=True, loc='upper left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        ax.imshow(plt.imread('/content/Bayesian-Crowd-Counting/datasets/frames/' + name[0] + '.jpg'))
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
    '''
