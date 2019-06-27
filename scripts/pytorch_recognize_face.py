# Loading in data


class Food11Dataset(Dataset):
    """Food 11 dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.index_file = pd.read_csv(
            csv_file)  # first column: image_name (i.e. 0809-personal.jpg)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.index_file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.index_file.iloc[idx, 0])  # get filepath to image
        image = io.imread(img_name)
        label = self.index_file.iloc[idx, 1]  # get the labels

        if self.transform:
            image = self.transform(image)

        return image, label
