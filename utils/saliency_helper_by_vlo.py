# coding=utf-8
# Copyright 2023 Viktor Loreth

"""Saliency helper library to compute and pre-process saliency heatmaps.
The executable part of this file is used to test the functions. For the actual use of the functions,
please import the functions from this file."""

# Integrated Gradients, Gradient Saliency, Guide Backpropagation are used to generate saliency maps

from captum.attr import IntegratedGradients, GuidedBackprop
from matplotlib.colors import LinearSegmentedColormap
import torch
import numpy as np
from PIL import Image
import os
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import json


def get_saliency_image(model, y, image, saliency_method):
    """generates saliency image.
        Args:
          model: model to compute saliency maps.
          y: the pre-softmax activation we want to assess attribution with respect to.
          image: float32 image tensor with size [1, None, None].
          saliency_method: string indicating saliency map type to generate.
        Returns:
          a saliency map and a smoothed saliency map.
        Raises:
          ValueError: if the saliency_method string does not match any included method
        """

    if saliency_method == "integrated_gradients":
        integrated_placeholder = IntegratedGradients(model)
        return integrated_placeholder.attribute(image, target=y)

    elif saliency_method == "guided_backprop":
        gb_placeholder = GuidedBackprop(model)
        return gb_placeholder.attribute(image, target=y)

    else:
        raise ValueError("No saliency method method matched. Verification of"
                         "input needed")


def generate_masks(mask: torch.Tensor, thresholds=None):
    """
    :param thresholds: thresholds to use for the saliency mask
    :param mask: saliency mask
    :param threshold: how much of the image should be blacked out (0.5 = 50%)
    :return: a image with the blocked out saliency mask
    """

    # convert mask to single channel
    mask = mask[0]
    mask_list = []

    # add constant 0.003 to each pixel, so it's positive
    mask = mask + 0.003

    # loop over all channels with a threshold until we achieve the desired threshold
    for threshold in thresholds:
        threshold_parameter = 1.01
        total_mask = torch.zeros((224, 224)).bool()
        k = 0
        threshold = threshold * 224 * 224  # maximum pixels blurred
        while (k < threshold - 0.01):  # 0.01 is a balancing term to avoid iteration errors
            threshold_parameter = threshold_parameter * 0.8
            # Loop through all channels with a threshold and blure those pixels. If the threshold is not reached,
            # increase the threshold

            for i in range(3):
                tmp_mask = mask[i] > threshold_parameter
                # add tmp_mask to total_mask

                total_mask += tmp_mask
                # convert total_mask to boolean
                total_mask = total_mask > 0
                k = total_mask.sum()
                if k > threshold:
                    # print(k/(224*224), threshold/(224*224))

                    break

        # save the total mask
        mask_list.append(total_mask)

    return mask_list, len(thresholds)


def apply_mask_to_image(image, mask: np.array):
    # convert True values to ImgNet mean
    # image = 3*224*224
    # mask = 224*224
    # if mask is true, replace the pixel with the mean
    # if mask is false, keep the pixel
    # return image
    image = image[0]
    image = image.permute(1, 2, 0)

    image[mask] = torch.tensor([0.485, 0.456, 0.406])

    image = image.permute(2, 0, 1)

    return image


def calculate_saliency_map(model, image_path, thresholds=None, cuda=False, return_mask=False, project_path=None):
    """
    :param model: model to compute saliency maps.
    :param image_path: path to the image
    :param thresholds: thresholds to use for the saliency mask
    :return: a image with the blocked out saliency mask
    """

    img = Image.open(project_path + '/' + image_path)
    mean = [0.485, 0.456, 0.406]

    # Transformer always stays the same
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda" if cuda == 'cuda' else "cpu")

    transformed_img = torch.unsqueeze(transform(img), 0)
    transformed_img = transformed_img.to(device)

    output = model(transformed_img)
    output = F.softmax(output, dim=1)
    _, pred_label_idx = torch.topk(output, 1)

    saliency_map = get_saliency_image(model, pred_label_idx, transformed_img, "integrated_gradients")

    masks, masks_len = generate_masks(saliency_map, thresholds=thresholds)

    if return_mask:
        return masks, masks_len
    else:
        # apply mask to image
        for i, mask in enumerate(masks):
            new_img = apply_mask_to_image(transformed_img.clone(), masks[i])
            # save new image np file

            new_img = new_img.permute(1, 2, 0)
            new_img = new_img.cpu().detach().numpy()
            # save new_img as jpeg file
            new_img = Image.fromarray((new_img * 255).astype(np.uint8))
            new_img.save(project_path + str(int(thresholds[i] * 100)) + '/' + image_path[:-5] + '.jpeg')


# Windows
project_path = r'C:\Users\Vik\Documents\4. Private\01. University\2022_Sem5\Intepretable_AI'
# Linux
# project_path = r'/home/viktorl/Intepretable_AI_PR_Loreth/'

if __name__ == '__main__':

    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    plot = True
    print("Running saliency Helper by vlo to test the functions")

    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.to(device)
    model.eval()

    mean = [0.485, 0.456, 0.406]

    # Transformer always stays the same
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load imagenet_labels
    with open(project_path + '/datasets/imagenet_class_index.json') as f:
        imagenet_labels = json.load(f)
    imagenet_labels = {int(k): v for k, v in imagenet_labels.items()}

    from PIL import Image

    img_path = r'/datasets/imagenet1000samples/n01491361_tiger_shark.JPEG'
    img = Image.open(project_path + img_path)

    transformed_img = transform(img)
    # send to cuda
    transformed_img = transformed_img.to(device)

    input_img = torch.unsqueeze(transformed_img, 0)

    orig_img = input_img.clone()
    # set model to evaluation mode and run img
    model.eval()
    output = model(input_img)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    if plot:
        print('Predicted:', pred_label_idx.item(), 'with score:', prediction_score.item())
        print('Predicted:', imagenet_labels[pred_label_idx.item()])

    import matplotlib.pyplot as plt

    # get saliency map
    saliency_map = get_saliency_image(model, pred_label_idx, input_img, "integrated_gradients")

    # set variable to plot
    if plot:
        # plot saliency map
        plt.imshow(saliency_map[0].permute(1, 2, 0))
        # plot pixel weight bar
        plt.colorbar()
        plt.show()

    # generate mask
    masks, masks_len = generate_masks(saliency_map, thresholds=[0.3, 0.5, 0.7])
    # display masks
    if plot:
        for i in range(masks_len):
            plt.imshow(masks[i])
            plt.show()

    # concatenate masks and save them
    if False:
        masks = torch.stack(masks)
        torch.save(masks, f'C:\\Users\\Vik\Documents\\4. Private\\01. University\\2022_Sem5\\Intepretable_AI\\masks.pt')
        print("File saved")
    ### Code is redundant and just for displaying the masks! ###

    if plot:
        # apply all masks to img
        img_list = []
        for i, mask in enumerate(masks):
            img_list.append(apply_mask_to_image(orig_img.clone(), masks[i]))

        orig_img = orig_img[0].permute(1, 2, 0)
        # plot masked img in shape 3*244*244
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(orig_img)
        axs[0, 0].set_title('Original Image')
        axs[0, 1].imshow(img_list[0].permute(1, 2, 0))
        axs[0, 1].set_title('Masked Image 30%')
        axs[1, 0].imshow(img_list[1].permute(1, 2, 0))
        axs[1, 0].set_title('Masked Image 50%')
        axs[1, 1].imshow(img_list[2].permute(1, 2, 0))
        axs[1, 1].set_title('Masked Image 70%')

        print("Green means the pixel is important and blocked out.")
# %%
