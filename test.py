import torch.utils.data
import os
import torch
import torch.utils.data
import numpy as np
import cv2
import utils
import transforms as T
from bert.modeling_bert import BertModel
from lib import segmentation


def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes


def evaluate(model, data_loader, dataset, bert_model, device,
             output_dir):
    """Evaluate the model and save predicted masks with overlays."""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    cum_I, cum_U = 0, 0
    header = 'Test:'

    # Track IoU for each class
    classwise_IU = {cat_id: {"I": 0, "U": 0} for cat_id in dataset.refer.Cats.keys()}

    with torch.no_grad():
        for idx, data in enumerate(metric_logger.log_every(data_loader, 100, header)):
            image, target, sentences, attentions = data

            # Get corresponding filename from dataset
            img_info = dataset.imgs[idx]  # Ensures correct mapping
            img_filename = img_info['file_name']

            # Move tensors to device
            image, target, sentences, attentions = (
                image.to(device), target.to(device), sentences.to(device), attentions.to(device)
            )

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().numpy()

            for j in range(sentences.size(-1)):
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                else:
                    output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

                output = output.cpu()
                output_mask = output.argmax(1).numpy()

                # Get category ID
                ref_id = dataset.ref_ids[idx]
                # print(dataset.refer.Refs.values())
                # print("Unique category IDs in dataset:", set([r['category_id'] for r in dataset.refer.Refs.values()]))
                # print("All categories available in dataset:", dataset.refer.Cats)
                # print("Reference categories:",
                #       [dataset.refer.Cats[r['category_id']] for r in dataset.refer.Refs.values()])
                # print("Unique predicted labels:", np.unique(output_mask))
                # print("Unique ground truth labels:", np.unique(target))
                # print("Model Output Shape:", output.shape)  # Should be [batch, num_classes, H, W]

                # import pdb
                # pdb.set_trace()

                category_id = dataset.refer.Refs[ref_id]['category_id']
                # print(f"[DEBUG] Image Index: {idx}, Ref ID: {ref_id}, Category ID: {category_id}")
                # print(category_id)

                # Compute IoU
                I, U = computeIoU(output_mask, target)
                mean_IoU.append(I / U if U != 0 else 0)
                cum_I += I
                cum_U += U

                # Update per-class IoU
                classwise_IU[category_id]["I"] += I
                classwise_IU[category_id]["U"] += U

                for n_eval_iou in range(len(eval_seg_iou_list)):
                    seg_correct[n_eval_iou] += (I / U >= eval_seg_iou_list[n_eval_iou])
                seg_total += 1

                # Save Predicted Masks with Overlays
                save_mask_with_overlay(image, output_mask[0], target[0], img_filename, output_dir)

            # Free memory
            del image, target, sentences, attentions, output, output_mask
            if bert_model is not None:
                del last_hidden_states, embedding

    # Print final IoU results
    print('\nFinal results:')
    print('Mean IoU is {:.2f}%'.format(np.mean(mean_IoU) * 100))

    for n_eval_iou in range(len(eval_seg_iou_list)):
        print(
            'precision@{} = {:.2f}%'.format(eval_seg_iou_list[n_eval_iou], seg_correct[n_eval_iou] * 100. / seg_total))

    print('Overall IoU = {:.2f}%'.format(cum_I * 100. / cum_U))

    # Print per-class IoU
    print("\nPer-Class IoU Results:")
    for class_id, values in classwise_IU.items():
        class_name = dataset.refer.Cats[class_id]  # Get class name
        I, U = values["I"], values["U"]
        iou = (I / U) * 100 if U > 0 else 0.0
        print(f"{class_name}: IoU = {iou:.2f}%")


def save_mask_with_overlay(image_tensor, pred_mask, gt_mask, img_filename, output_dir):
    """Overlay predicted and ground truth masks on original image and save with proper resizing and denormalization."""

    # Convert tensor to numpy (CHW -> HWC)
    image = image_tensor.cpu().numpy()[0].transpose(1, 2, 0)

    # Reverse normalization (denormalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image * std + mean)  # Convert back to original pixel values
    image = np.clip(image * 255, 0, 255).astype(np.uint8)  # Convert to uint8

    # Resize masks to match image size (Ensures alignment)
    pred_mask = cv2.resize(pred_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_mask = cv2.resize(gt_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert masks to color
    pred_color = np.zeros_like(image)
    pred_color[:, :, 1] = pred_mask * 255  # Green for predicted mask

    gt_color = np.zeros_like(image)
    gt_color[:, :, 1] = gt_mask * 255  # Green for ground truth mask

    # Blend images with transparency
    alpha = 0.5  # Transparency factor
    overlay_pred = cv2.addWeighted(image, 1, pred_color, alpha, 0)
    overlay_gt = cv2.addWeighted(image, 1, gt_color, alpha, 0)

    overlay_pred = cv2.cvtColor(np.array(overlay_pred), cv2.COLOR_RGB2BGR)
    overlay_gt = cv2.cvtColor(np.array(overlay_gt), cv2.COLOR_RGB2BGR)

    # Save images
    pred_output_path = os.path.join(output_dir, f"overlay_pred_{img_filename}")
    gt_output_path = os.path.join(output_dir, f"overlay_gt_{img_filename}")

    cv2.imwrite(pred_output_path, overlay_pred)
    cv2.imwrite(gt_output_path, overlay_gt)

    # print(f"Saved: {pred_output_path}, {gt_output_path}")  # Debugging line


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):
    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained='', args=args)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    single_model.load_state_dict(checkpoint['model'], strict=False)
    model = single_model.to(device)

    if args.model != 'lavt_one':
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    evaluate(model, data_loader_test, dataset_test, bert_model, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)