import wget
import os
import skimage.io as io

# dict
DATASETS={
    "lenna_png":{"url" : "https://murhyimgur.oss-cn-beijing.aliyuncs.com/cvpr/lena_std.png",
            "format" : "png",
            "folder" : "lenna/",
            "download" : True,
            },
    "lenna_jpg":{"url":"https://murhyimgur.oss-cn-beijing.aliyuncs.com/cvpr/lena_std.jpg",
            "format":"jpg",
            "folder" : "lenna/",
            "download" : True,
            },
    "lenna_tif":{"url":"https://murhyimgur.oss-cn-beijing.aliyuncs.com/cvpr/lena_std.tif",
            "format":"tif",
            "folder" : "lenna/",
            "download" : True,
            } 
}

def get_dataset(dataset_name='lenna_jpg', target_folder='../datasets/', datasets=DATASETS):

    # judge dataset exit or not
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset in unknown.".format(dataset_name))
    
    dataset = datasets[dataset_name]
    folder = target_folder + dataset.get("folder", 'img/')
    if dataset.get("download"):
        if not os.path.isdir(folder):
            os.makedirs(folder)
        filename = wget.filename_from_url(dataset["url"])
        if not os.path.exists(folder + filename):
            wget.download(dataset["url"], out=folder + filename)
    else:
        print("Warninig: {} is not downloadable!".format(dataset_name))

    img = io.imread(folder + filename)
    return img
