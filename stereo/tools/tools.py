import os


def make_dirs(output_path: str) -> None:
    '''
    Determine whether the file exists, create it if it does not exist

    :param output_path: file path
    :return: None
    '''

    if output_path.startswith('/'):
        file_path = output_path.rsplit('/', 1)[0]
    else:
        tail_file_path = output_path.rsplit('/', 1)[0].replace('.', '')
        file_path = '/'.join([os.getcwd(), tail_file_path])

    if not os.path.exists(file_path):
        os.makedirs(file_path)
