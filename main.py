import logging
from processing.data_loader import DataLoader


def main():
    logging.basicConfig(
        filename='occAccuracies1.log',
        format='%(asctime)s: %(message)s',
        level=logging.INFO
    )
    loader = DataLoader()
    loader.add_folder_to_model("data/cleaned-data-2021-03-14")
    loader.fit()
    loader.store_processed_files()

    return


if __name__ == "__main__":
    main()
