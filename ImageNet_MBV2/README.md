## Implementation of VKDNW on the MobileNetV2 search space
- Prepare the ImageNet dataset in the directory `/dataset/ILSVRC2012`, or manually change the directory specified in `./Dataloader/__init__.py` (Lines 33-34).
- Run the scripts in the `scripts` folder. For example:

## Credit
- The code is modified from [AZ-NAS](https://github.com/cvlab-yonsei/AZ-NAS)

## Change Notes
- `evolutionary_search_vkdnw.py`: Modified from [`evolutionary_search_az.py`](https://github.com/cvlab-yonsei/AZ-NAS/blob/master/ImageNet_MBV2/evolution_search_az.py)
> *  Implement an evolutionary search algorithm for VKDNW