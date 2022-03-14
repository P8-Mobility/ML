import logging

from occ import OCC


def main():
    logging.basicConfig(
        filename='occAccuracies1.log',
        format='%(asctime)s: %(message)s',
        level=logging.INFO
    )
    occ = OCC(False)
    occ.run()
    return


if __name__ == "__main__":
    main()
