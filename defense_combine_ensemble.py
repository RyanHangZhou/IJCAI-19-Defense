import argparse

import os
import io
import random
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import models
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True, help='Input directory.')
parser.add_argument('--output_file', required=True, help='Output file.')
parser.add_argument('--checkpoint_1', required=True, help='checkpoint path')
parser.add_argument('--checkpoint_2', required=True, help='checkpoint path')
parser.add_argument('--checkpoint_3', required=True, help='checkpoint path')
parser.add_argument('--checkpoint_4', required=True, help='checkpoint path')
parser.add_argument('--checkpoint_5', required=True, help='checkpoint path')
parser.add_argument('--checkpoint_6', required=True, help='checkpoint path')
parser.add_argument('--checkpoint_7', required=True, help='checkpoint path')
parser.add_argument('--checkpoint_8', required=True, help='checkpoint path')
parser.add_argument('--checkpoint_9', required=True, help='checkpoint path')
parser.add_argument('--checkpoint_10', required=True, help='checkpoint path')
parser.add_argument('--batch_size', default=4, type=int, help="Batch size.")
parser.add_argument('--workers', default=4, type=int, help='number of workers')

IMG_EXTENSIONS = ('jpg', 'jpeg', 'png', 'ppm',
                  'bmp', 'pgm', 'tif', 'tiff', 'webp')


class ImageFolderDataset(data.Dataset):
    def __init__(self, root, transform=None):
        super().__init__()

        self.transform = transform
        self.file_list = []

        for f in os.listdir(root):
            fullpath = os.path.join(root, f)
            if os.path.isfile(fullpath) \
                    and fullpath.rsplit(".", 1)[-1] in IMG_EXTENSIONS:
                self.file_list.append(fullpath)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        fullpath = self.file_list[index]
        img = Image.open(fullpath)
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fullpath


def main():
    args = parser.parse_args()

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.500, 0.500, 0.500],
                                     std=[0.500, 0.500, 0.500])
    def preproc(img):
        with io.BytesIO() as fout:
            img.save(fout, format="JPEG", quality=25)
            fout.seek(0)
            img = Image.open(fout)
            return img.convert("RGB")

    transform = transforms.Compose([
        

        transforms.Resize(310),
        transforms.Lambda(preproc),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize
    ])

    dataset = ImageFolderDataset(args.input_dir, transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    ckpt_1 = torch.load(args.checkpoint_1)
    arch_1 = ckpt_1["arch"]
    model_1 = getattr(models, arch_1)(num_classes=110, pretrained=None)
    model_1 = torch.nn.DataParallel(model_1)
        #bhy
    #state_dict =ckpt_1['state_dict']
    #from collections import OrderedDict
    #new_state_dict = OrderedDict()

    #for k, v in state_dict.items():
    #  if 'module' not in k:
    #      k = 'module.'+k
    #  else:
    #      k = k.replace('features.module.', 'module.features.')
    #  new_state_dict[k]=v
    #model_1.load_state_dict(new_state_dict)
    model_1.load_state_dict(ckpt_1["state_dict"])
    model_1.cuda()
    model_1.eval()
    
    ckpt_2 = torch.load(args.checkpoint_2)
    arch_2 = ckpt_2["arch"]
    model_2 = getattr(models, arch_2)(num_classes=110, pretrained=None)
    model_2 = torch.nn.DataParallel(model_2)
    model_2.load_state_dict(ckpt_2["state_dict"])
    model_2.cuda()
    model_2.eval()

    ckpt_3 = torch.load(args.checkpoint_3)
    arch_3 = ckpt_3["arch"]
    model_3 = getattr(models, arch_3)(num_classes=110, pretrained=None)
    model_3 = torch.nn.DataParallel(model_3)
    model_3.load_state_dict(ckpt_3["state_dict"])
    model_3.cuda()
    model_3.eval()

    ckpt_4 = torch.load(args.checkpoint_4)
    arch_4 = ckpt_4["arch"]
    model_4 = getattr(models, arch_4)(num_classes=110, pretrained=None)
    model_4 = torch.nn.DataParallel(model_4)
    model_4.load_state_dict(ckpt_4["state_dict"])
    model_4.cuda()
    model_4.eval()
    
    ckpt_5 = torch.load(args.checkpoint_5)
    arch_5 = ckpt_5["arch"]
    model_5 = getattr(models, arch_5)(num_classes=110, pretrained=None)
    model_5 = torch.nn.DataParallel(model_5)
    model_5.load_state_dict(ckpt_5["state_dict"])
    model_5.cuda()
    model_5.eval()
    
    ckpt_6 = torch.load(args.checkpoint_6)
    arch_6 = ckpt_6["arch"]
    model_6 = getattr(models, arch_6)(num_classes=110, pretrained=None)
    model_6 = torch.nn.DataParallel(model_6)
    model_6.load_state_dict(ckpt_6["state_dict"])
    model_6.cuda()
    model_6.eval()

    ckpt_7 = torch.load(args.checkpoint_7)
    arch_7 = ckpt_7["arch"]
    model_7 = getattr(models, arch_7)(num_classes=110, pretrained=None)
    model_7 = torch.nn.DataParallel(model_7)
    model_7.load_state_dict(ckpt_7["state_dict"])
    model_7.cuda()
    model_7.eval()

    ckpt_8 = torch.load(args.checkpoint_8)
    arch_8 = ckpt_8["arch"]
    model_8 = getattr(models, arch_8)(num_classes=110, pretrained=None)
    model_8 = torch.nn.DataParallel(model_8)
    model_8.load_state_dict(ckpt_8["state_dict"])
    model_8.cuda()
    model_8.eval()

    ckpt_9 = torch.load(args.checkpoint_6)
    arch_9 = ckpt_9["arch"]
    model_9 = getattr(models, arch_9)(num_classes=110, pretrained=None)
    model_9 = torch.nn.DataParallel(model_9)
    model_9.load_state_dict(ckpt_9["state_dict"])
    model_9.cuda()
    model_9.eval()

    ckpt_10 = torch.load(args.checkpoint_10)
    arch_10 = ckpt_10["arch"]
    model_10 = getattr(models, arch_10)(num_classes=110, pretrained=None)
    model_10 = torch.nn.DataParallel(model_10)
    model_10.load_state_dict(ckpt_10["state_dict"])
    model_10.cuda()
    model_10.eval()    

    all_results = []
    all_files = []
    with torch.no_grad(), \
         open(args.output_file, "w") as fout:
        for i, (input, file_list) in enumerate(loader):
            input = input.cuda(non_blocking=True)
            output_1 = model_1(input)
            output_2 = model_2(input)
            output_3 = model_3(input)
            output_4 = model_4(input)
            output_5 = model_5(input)
            output_6 = model_6(input)
            output_7 = model_7(input)
            output_8 = model_8(input)
            output_9 = model_9(input)
            output_10 = model_10(input)

            for ks in [1, 3, 5, 7, 9, 11]:
                w = torch.rand((3, 1, ks*2+1, ks*2+1)).cuda()
                w = w/(w.sum([1, 2, 3]).view([3, 1, 1, 1]))
                temp = F.conv2d(input, w, padding=ks, groups=3)

                output_1 += model_1(temp)/2
                output_2 += model_2(temp)/2
                output_3 += model_3(temp)/2
                output_4 += model_4(temp)/2
                output_5 += model_5(temp)/2
                output_6 += model_6(temp)/2
                output_7 += model_7(temp)/2
                output_8 += model_8(temp)/2
                output_9 += model_9(temp)/2
                output_10 += model_10(temp)/2


            # all_results.append(output_1)
            # all_files.extend(file_list)
            _, predicted = torch.max(output_1.data+output_2.data+output_3.data+output_4.data+output_5.data+output_6.data+output_7.data+output_8.data+output_9.data+output_10.data, 1)
            for fullpath, pred in zip(file_list, predicted):
                print("%s,%d" % (os.path.basename(fullpath), pred), file=fout)


if __name__ == '__main__':
    main()
