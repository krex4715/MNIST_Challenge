import argparse
import os
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import numpy as np
from LeNet5 import LeNet
from sklearn.metrics import accuracy_score
import tensorflow as tf
import torch


class ImageDataset(Dataset):
    """ Image shape: 28x28x3 """

    def __init__(self, root_dir, fmt=':06d', extension='.png'):
        self.root_dir = root_dir
        self.fmtstr = '{' + fmt + '}' + extension

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.fmtstr.format(idx)
        img_path = os.path.join(self.root_dir, img_name)
        img = plt.imread(img_path)
        data = torch.from_numpy(img)
        return data


def torch_to_numpy_convert(data_loader):
  a = []
  for x in data_loader:
    a.append(np.array(x.reshape(28,28,3)))
  return np.array(a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2022 DL Term Project #1')
    parser.add_argument('--dataset', default='./test_img/', help='image dataset directory')
    parser.add_argument('--batch-size', default=1, help='test loader batch size')

    args = parser.parse_args()

    # instantiate model
    model = tf.keras.models.load_model('./model_params/LeNet5/LeNet_5.h5')
    # load dataset in test image folder
    test_data = ImageDataset(args.dataset)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    a = torch_to_numpy_convert(test_loader)
    predicts = model.predict(a)

    preds=[]
    for i in range(len(predicts)):
        preds.append(predicts[i].argmax())

    with open('result.txt', 'w') as f:
        f.writelines('\n'.join(map(str, preds)))

    true = [7,2,1,0,4,1,4,9,5,9,0,6,9,0,1,5,9,7,3,4,9,6,6,5,4,0,7,4]
    print('True Label is....', true)
    print('Prediction is....',preds)

    print('Accuracy : ',end='')
    print(accuracy_score(true, preds))
