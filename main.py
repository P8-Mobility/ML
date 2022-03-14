import logging
from processing.data_loader import DataLoader


def main():
    logging.basicConfig(
        filename='occAccuracies1.log',
        format='%(asctime)s: %(message)s',
        level=logging.INFO
    )
    loader = DataLoader()
    loader.add_folder_to_model("data/train")
    loader.fit()
    loader.save_files("processed")

    return


if __name__ == "__main__":
    main()
