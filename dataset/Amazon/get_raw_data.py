import os

import requests
import os.path as osp
from tqdm import tqdm

base_url = "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/{}"
base_name = "{}_5.json.gz"

def category_name_to_filename(catetory_name: str) -> str:
    return base_name.format(catetory_name.replace(", ", "_").replace(" ", "_"))


def download(file_path: str, filename: str) -> bool:
    download_url = base_url.format(filename)
    print("Download: {}".format(download_url))
    try:
        r = requests.get(download_url, stream=True)
        file_size = r.headers['Content-length']
        pbar = tqdm(total=int(file_size), unit='B', unit_scale=True)
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(1024)
            pbar.close()
        return True
    except Exception as e:
        raise ("Download file {} failed".format(file_path), e)

def get_raw_data_path(category: str, target_path: str, redownload: bool=False)->str:
    filename = category_name_to_filename(category)
    file_path = osp.join(target_path, filename)
    if not osp.exists(file_path) or redownload:
        ok = download(file_path, filename)
        if ok:
            print("{} saved at {}".format(filename, file_path))
            return file_path
    else:
        print("File {} already exists in: {}".format(filename, file_path))
        return file_path




if __name__ == "__main__":
    categories = ['Books', 'Digital Music', 'Movies and TV', 'Office Products', 'Video Games']
    target_path = os.path.split(os.path.realpath(__file__))[0]
    replace = True
    for c in categories:
        get_raw_data_path(c, target_path)



