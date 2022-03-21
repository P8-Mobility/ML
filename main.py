from processing.data_loader import DataLoader


def main():
    loader = DataLoader()
    loader.change_setting("trim_threshold", 20)
    loader.change_setting("scale_length", False)
    loader.add_folder_to_model("data/cleaned-data-2021-03-14")
    loader.fit()
    loader.store_processed_files()

    return


if __name__ == "__main__":
    main()
