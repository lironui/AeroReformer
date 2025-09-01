import os
import sys  # Not used directly here, but often in main scripts
import torch.utils.data as data
import torch
from torchvision import transforms  # Not used directly here, but often in main scripts
from torch.autograd import Variable  # Not used directly here, but often in main scripts
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random
from pathlib import Path  # For easier path manipulation

from bert.tokenization_bert import BertTokenizer

import h5py  # Not used in the snippet, but often part of REFER
from refer.refer import REFER  # Assuming this is your REFER API


# from args import get_parser # Assuming args are passed in

# # Dataset configuration initialization (Example, if args were defined here)
# # parser = get_parser()
# # args = parser.parse_args([]) # Provide default or empty list for testing

def add_random_boxes(img, min_num=20, max_num=60, size=32):
    # Ensure img is mutable and in a format we can work with (numpy array)
    if isinstance(img, Image.Image):
        img_array = np.array(img).copy()
    else:  # Assuming it might already be a numpy array
        img_array = np.asarray(img).copy()

    if img_array.ndim == 2:  # Grayscale image
        img_array = np.stack((img_array,) * 3, axis=-1)  # Convert to 3-channel
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Convert to RGB

    h_box, w_box = size, size
    img_h, img_w = img_array.shape[0], img_array.shape[1]  # Use actual image dimensions

    if img_h <= h_box or img_w <= w_box:  # If image is smaller than box, don't add boxes
        if isinstance(img, Image.Image):
            return Image.fromarray(img_array.astype('uint8'), 'RGB')
        return img_array

    num_boxes = random.randint(min_num, max_num)
    for _ in range(num_boxes):
        y = random.randint(0, img_h - h_box)
        x = random.randint(0, img_w - w_box)
        img_array[y:y + h_box, x:x + w_box, :] = 0  # Set all channels to 0 for black boxes

    return Image.fromarray(img_array.astype('uint8'), 'RGB')


class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []  # You might want to populate this from self.refer.Cats
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.split = split
        self.args = args

        print(f"Initializing ReferDataset for split: {self.split}")
        print(f"  Refer data root: {args.refer_data_root}")
        print(f"  Dataset: {args.dataset}, SplitBy: {args.splitBy}")

        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)

        self.max_tokens = args.max_seq_length

        # Get ref_ids for the current split. These are the primary keys for our items.
        # The order of ref_ids from getRefIds is assumed to be canonical for this split.
        ref_ids_for_split = self.refer.getRefIds(split=self.split)
        if not ref_ids_for_split:
            raise ValueError(f"No ref_ids found for split '{self.split}'. Check dataset and splitBy parameters.")

        # For each ref_id, get its corresponding image_id.
        # It's assumed getImgIds(ref_ids) returns a list of image_ids where img_ids[i] corresponds to ref_ids[i].
        # And for our preprocessed data, each ref_id maps to exactly one image_id.
        img_ids_for_refs = [self.refer.getImgIds(ref_id)[0] for ref_id in ref_ids_for_split]

        all_imgs_map = self.refer.Imgs  # This is a dictionary: img_id -> img_info

        # self.imgs will be a list of image_info dictionaries, ordered according to ref_ids_for_split.
        self.imgs = []
        for i_idx, img_id in enumerate(img_ids_for_refs):
            if img_id not in all_imgs_map:
                raise ValueError(
                    f"Consistency error: img_id {img_id} (for ref_id {ref_ids_for_split[i_idx]}) not in self.refer.Imgs map.")
            self.imgs.append(all_imgs_map[img_id])

        self.ref_ids = ref_ids_for_split  # Store the ordered list of ref_ids

        # Sanity check: lengths must match
        if len(self.imgs) != len(self.ref_ids):
            raise ValueError(
                f"Mismatch in lengths of self.imgs ({len(self.imgs)}) and self.ref_ids ({len(self.ref_ids)}). This indicates an issue in data alignment.")

        print(f"  Loaded {len(self.ref_ids)} references for split '{self.split}'.")

        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.eval_mode = eval_mode

        for i, r_id in enumerate(self.ref_ids):  # Iterate using the canonical order of ref_ids
            ref = self.refer.Refs[r_id]  # Get reference details directly by ref_id

            sentences_for_this_ref = []
            attentions_for_this_ref = []

            if not ref['sentences']:
                print(f"Warning: No sentences found for ref_id {r_id}. Skipping tokenization for this ref.")
                # Add placeholder empty tensors if necessary, or handle downstream
                # For now, let's ensure the lists self.input_ids and self.attention_masks
                # still get an entry for this ref_id to maintain length consistency.
                self.input_ids.append([])
                self.attention_masks.append([])
                continue

            for sent_data in ref['sentences']:  # el is sentence_data dict
                sentence_raw = sent_data['raw']

                # Tokenize
                encoded_dict = self.tokenizer.encode_plus(
                    sentence_raw,
                    add_special_tokens=True,
                    max_length=self.max_tokens,
                    padding='max_length',  # Pad to max_length
                    truncation=True,  # Truncate to max_length
                    return_attention_mask=True,
                    return_tensors='pt',  # Return PyTorch tensors
                )

                # encoded_dict['input_ids'] is [1, max_tokens], .squeeze(0) to make it [max_tokens]
                # Storing as list of tensors, will be stacked in __getitem__ if needed or handled per sentence
                sentences_for_this_ref.append(encoded_dict['input_ids'])  # Shape [1, max_tokens]
                attentions_for_this_ref.append(encoded_dict['attention_mask'])  # Shape [1, max_tokens]

            self.input_ids.append(sentences_for_this_ref)  # List of lists of tensors
            self.attention_masks.append(attentions_for_this_ref)  # List of lists of tensors

        # Final length check for token lists
        if len(self.input_ids) != len(self.ref_ids) or len(self.attention_masks) != len(self.ref_ids):
            raise ValueError(
                "Mismatch in lengths after tokenization. Ensure all ref_ids have corresponding token entries.")

    def get_classes(self):
        # Populate self.classes from self.refer.Cats if not already done
        if not self.classes and hasattr(self.refer, 'Cats'):
            self.classes = [cat['name'] for cat_id, cat in sorted(self.refer.Cats.items())]
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        # Use the co-indexed lists established in __init__
        this_ref_id = self.ref_ids[index]
        this_img_metadata = self.imgs[index]  # Contains 'file_name', 'id' (image_id), etc.

        # Load image using file_name from the consistent self.imgs list
        image_file_path = os.path.join(self.refer.IMAGE_DIR, this_img_metadata['file_name'])

        try:
            img = Image.open(image_file_path).convert('RGB')  # Ensure RGB

        except FileNotFoundError:
            print(f"FATAL: Image file not found at {image_file_path} for ref_id {this_ref_id}, index {index}.")
            raise
        except Exception as e:
            print(f"FATAL: Error loading image {image_file_path}: {e}")
            raise

        # Apply random boxes augmentation if in training mode and for selected images
        # Note: self.imacontinueges_to_mask was not defined in the provided __init__.
        # If you need this, ensure it's initialized (e.g., random sample of self.ref_ids).
        # For now, I'll comment it out or assume it's handled if args.apply_random_boxes is a new arg.
        # if self.split == 'train' and hasattr(self, 'images_to_mask') and this_ref_id in self.images_to_mask:
        # if self.split == 'train' and getattr(self.args, 'apply_random_boxes', False): # Example: control via args
        #     img = add_random_boxes(img)

        # Load reference details for this_ref_id
        # self.refer.Refs is a map: ref_id -> ref_info dict
        if this_ref_id not in self.refer.Refs:
            raise ValueError(f"FATAL: ref_id {this_ref_id} (at index {index}) not found in self.refer.Refs.")
        current_ref_obj = self.refer.Refs[this_ref_id]  # The ref object (dict)

        # Load the mask for this specific reference object.
        # getMaskxml MUST use current_ref_obj['ann_id'] to fetch the specific segmentation.
        try:
            # Pass the whole ref object, as getMask might need 'ann_id' or other fields
            mask_data = self.refer.getMaskxml(current_ref_obj)
            ref_mask_np = np.array(mask_data['mask'])
        except Exception as e:
            print(
                f"FATAL: Error getting mask for ref_id {this_ref_id} (ann_id {current_ref_obj.get('ann_id', 'N/A')}, img_file {this_img_metadata['file_name']}): {e}")
            raise

        # Convert mask to binary (0 or 1)
        annot_np = np.zeros(ref_mask_np.shape, dtype=np.uint8)
        # The condition for binarization might depend on how masks are stored in your XML/JSON
        if self.args.dataset == 'refuavid':  # Assuming RefUAVid uses 1 for foreground in its raw mask values
            annot_np[ref_mask_np == 1] = 1
        else:  # For other datasets, or if 255 is used for foreground
            # Check unique values in ref_mask_np if unsure: print(np.unique(ref_mask_np))
            annot_np[ref_mask_np > 0] = 1  # A more general case: any non-zero is foreground

        target_mask = Image.fromarray(annot_np, mode="P")  # "P" for 8-bit pixels, palette (0 and 1)

        # Apply image and target mask transformations
        if self.image_transforms is not None:
            # The transform should handle both image and mask
            img_tensor, target_tensor = self.image_transforms(img, target_mask)
        else:
            # Fallback basic transformation if none provided
            img_tensor = TF.to_tensor(img)
            target_tensor = TF.to_tensor(
                target_mask)  # Converts P mode (0-1) to [1, H, W] tensor with values 0.0 or 1.0

        # Handle sentences for the current reference (self.ref_ids[index])
        # self.input_ids[index] is a list of sentence tensors for self.ref_ids[index]
        current_ref_input_ids = self.input_ids[index]
        current_ref_attention_masks = self.attention_masks[index]

        if not current_ref_input_ids:  # Handle case of no sentences for a ref
            # This case should be rare if data is clean.
            # Return placeholder or skip? For now, create dummy tensors.
            # print(f"Warning: No tokenized sentences for ref_id {this_ref_id} at index {index}. Using dummy tensors.")
            # Assuming downstream can handle zero sentences or this item is filtered out by collate_fn
            # A single dummy sentence
            dummy_input_ids = torch.zeros((1, self.max_tokens), dtype=torch.long)
            dummy_attention_mask = torch.zeros((1, self.max_tokens), dtype=torch.long)
            if self.eval_mode:  # Need to match [1, max_tokens, num_sentences]
                tensor_embeddings = dummy_input_ids.unsqueeze(-1)
                attention_mask = dummy_attention_mask.unsqueeze(-1)
            else:
                tensor_embeddings = dummy_input_ids
                attention_mask = dummy_attention_mask

            return img_tensor, target_tensor, tensor_embeddings, attention_mask

        if self.eval_mode:
            # Concatenate all sentences for this reference along a new dimension
            # Each sentence tensor in current_ref_input_ids is [1, max_tokens]
            # Stack them to be [num_sentences, max_tokens], then permute.
            tensor_embeddings = torch.cat(current_ref_input_ids, dim=0).permute(1, 0).unsqueeze(
                0)  # -> [1, max_tokens, num_sentences]
            attention_mask = torch.cat(current_ref_attention_masks, dim=0).permute(1, 0).unsqueeze(
                0)  # -> [1, max_tokens, num_sentences]
        else:  # Training mode, pick one sentence randomly
            choice_sent_idx = np.random.choice(len(current_ref_input_ids))
            tensor_embeddings = current_ref_input_ids[choice_sent_idx]  # Shape [1, max_tokens]
            attention_mask = current_ref_attention_masks[choice_sent_idx]  # Shape [1, max_tokens]

        # import pdb;
        # pdb.set_trace()

        return img_tensor, target_tensor, tensor_embeddings, attention_mask
