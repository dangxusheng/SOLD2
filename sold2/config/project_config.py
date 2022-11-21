"""
Project configurations.
"""
import os


class Config(object):
    """ Datasets and experiments folders for the whole project. """
    #####################
    ## Dataset setting ##
    #####################
    DATASET_ROOT = os.getenv("DATASET_ROOT", "./datasets/")  # TODO: path to your datasets folder
    if not os.path.exists(DATASET_ROOT):
        os.makedirs(DATASET_ROOT)
    
    # Synthetic shape dataset
    synthetic_dataroot = os.path.join(DATASET_ROOT, "synthetic_shapes")
    synthetic_cache_path = os.path.join(DATASET_ROOT, "synthetic_shapes")
    if not os.path.exists(synthetic_dataroot):
        os.makedirs(synthetic_dataroot)
    
    # Exported predictions dataset
    export_dataroot = os.path.join(DATASET_ROOT, "export_datasets")
    export_cache_path = os.path.join(DATASET_ROOT, "export_datasets")
    if not os.path.exists(export_dataroot):
        os.makedirs(export_dataroot)
    
    # Wireframe dataset
    # wireframe_dataroot = os.path.join(DATASET_ROOT, "wireframe")
    wireframe_dataroot = '/home/sunnypc/dangxs/datasets/wireframe/v1.1'
    wireframe_cache_path = os.path.join(DATASET_ROOT, "wireframe")

    # Holicity dataset
    holicity_dataroot = os.path.join(DATASET_ROOT, "Holicity")
    holicity_cache_path = os.path.join(DATASET_ROOT, "Holicity")


    # Tray dataset
    # tray_dataroot = os.path.join(DATASET_ROOT, "Tray_crop")
    tray_dataroot = '/home/sunnypc/dangxs/datasets/Tray_Dataset/20220816_crop_img'
    tray_cache_path = os.path.join(DATASET_ROOT, "Tray_crop")


    # YorkUrban dataset
    yorkUrban_dataroot = '/home/sunnypc/dangxs/datasets/YorkUrbanDB'
    yorkUrban_cache_path = os.path.join(DATASET_ROOT, "YorkUrban")


    ########################
    ## Experiment Setting ##
    ########################
    EXP_PATH = os.getenv("EXP_PATH", "./experiments/")  # TODO: path to your experiments folder
    if not os.path.exists(EXP_PATH):
        os.makedirs(EXP_PATH)
