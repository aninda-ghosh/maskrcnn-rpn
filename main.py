import torch
import torch.utils.data
import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights

from engine import train_one_epoch, evaluate
import utils
import transforms as T


root_path = '/home/aghosh57/Kerner-Lab/exploration/all_dataset/'


class ParcelDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        with open(self.root+'data.txt') as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
            lines = [x.split(',') for x in lines]

        # Create separate lists for image, mask, and geojson file paths
        img_path = []
        mask_path = []
        geojson_path = []

        for i in range(len(lines)):
            img_path.append(self.root+lines[i][0])
            mask_path.append(self.root+lines[i][1])
            geojson_path.append(self.root+lines[i][2])

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = img_path
        self.masks = mask_path

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = np.array(boxes, dtype=np.float32)

        labels = np.ones((num_objs,), dtype=np.int64)        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = np.zeros((num_objs,), dtype=np.int64)

        temptargets = {}
        temptargets["boxes"] = boxes
        temptargets["labels"] = labels
        temptargets["masks"] = masks
        temptargets["image_id"] = idx
        temptargets["area"] = area
        temptargets["iscrowd"] = iscrowd


        #Remove the entries with area 0
        target = {}
        target["boxes"] = []
        target["labels"] = []
        target["masks"] = []
        target["image_id"] = idx
        target["area"] = []
        target["iscrowd"] = []

        for i in range(len(temptargets["area"])):
            if temptargets["area"][i] != 0:
                target["boxes"].append(temptargets["boxes"][i])
                target["labels"].append(temptargets["labels"][i])
                target["masks"].append(temptargets["masks"][i])
                target["area"].append(temptargets["area"][i])
                target["iscrowd"].append(temptargets["iscrowd"][i])

        target["boxes"] = torch.as_tensor(np.array(target["boxes"], dtype=np.float32))
        target["labels"] = torch.as_tensor(np.array(target["labels"], dtype=np.int64))
        target["masks"] = torch.as_tensor(np.array(target["masks"], dtype=np.uint8))
        target["image_id"] = torch.as_tensor(target["image_id"])
        target["area"] = torch.as_tensor(np.array(target["area"], dtype=np.float32))
        target["iscrowd"] = torch.as_tensor(np.array(target["iscrowd"], dtype=np.int64))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    

      
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)




def main():
    # use our dataset and defined transformations
    dataset = ParcelDataset(root_path, get_transform(train=True))
    dataset_test = ParcelDataset(root_path, get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    

    torch.cuda.empty_cache()


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
    

    # let's train it for 10 epochs

    num_epochs = 30

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        
        # update the learning rate
        lr_scheduler.step()
        
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        torch.cuda.empty_cache()

    # save model
    torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == '__main__':
    main()