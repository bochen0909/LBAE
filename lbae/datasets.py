from __future__ import print_function
__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '1.8'
__status__ = "Research"
__date__ = "2/1/2020"
__license__= "MIT License"

from torchvision.datasets import MNIST



def corrupt(x, corrupt_method, corrupt_args):
    """
        Disabled
    """
    return x



class MNISTEx(MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 corrupt_method='noise', corrupt_args=[0.4]):
        super().__init__(root, train, transform, target_transform, download)
        self.corrupt_method = corrupt_method
        self.corrupt_args = corrupt_args
        
    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        # Corrupt images
        corrupted = corrupt(image, self.corrupt_method, self.corrupt_args)
        return image, target, corrupted


if __name__ == "__main__":
    print("NOT AN EXECUTABLE!")

