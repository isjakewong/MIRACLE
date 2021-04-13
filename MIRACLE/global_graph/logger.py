import logging


def get_logger(dataset_name):
    filename = dataset_name
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # output to file
    fh = logging.FileHandler('{}.log'.format(filename), mode='w')
    fh.setLevel(logging.INFO)
    # output to screen
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger