"""Download or generate datasets"""
import os
import urllib
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """Custom progress bar for downloader"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

class MakeData:
    """Downloading or generating data"""
    @staticmethod
    def download_data(data_uri:str, save_path:str="/home/kemalaraz/Desktop/BERTNER/data/raw"):
        assert os.path.isdir(save_path), "Save path must be a directory"
        assert os.path.exists(os.path.join(save_path, data_uri.split("/")[-1])), "File exists"
        with DownloadProgressBar(unit='B', unit_scale=True,
                                miniters=1, desc=data_uri.split('/')[-1]) as t:
            urllib.request.urlretrieve(data_uri, filename=os.path.join(save_path, data_uri.split("/")[-1]), reporthook=t.update_to)

    def generate(self):
        """Generates syntetic data"""
        raise NotImplementedError