import os

from ..log_manager import logger


def _download(url: str, dir_str: str = "./stereopy_data/", file_name: str = None):
    # in order to return at first when the runner with no network
    if file_name and os.path.isfile(dir_str + file_name):
        return dir_str + file_name

    from tqdm import tqdm
    from urllib.request import urlopen, Request
    from urllib.error import URLError

    block_size = 1024 * 8
    block_num = 0

    try:
        req = Request(url, headers={"User-agent": "stereopy-user"})

        try:
            open_url = urlopen(req)
        except URLError:
            logger.warning(
                'Failed to open the url with default certificates, trying with certifi.'
            )

            from certifi import where
            from ssl import create_default_context
            open_url = urlopen(req, context=create_default_context(cafile=where()))

        with open_url as resp:
            if file_name is None:
                content_disposition = resp.info().get("content-disposition", None)
                if content_disposition:
                    file_name = content_disposition.split(';')[1].split('=')[1].replace('\"', '')
                else:
                    raise Exception('remote file not exists')

            from pathlib import Path
            if not dir_str:
                # TODO: use config.data_dir, now will download to site-package, this is bad
                dir_str = "./stereopy_data/"

            path = Path(dir_str)
            if not path.is_dir():
                path.mkdir()

            path = Path(path.__str__() + '/' + file_name)
            if path.is_file():
                return path.__str__()

            total = resp.info().get("content-length", None)
            total_mb = str('%.4f') % (int(total) / 1024 / 1024)
            logger.info(f'You are starting to download a {total_mb} MB file named {file_name}...')

            with tqdm(
                    unit="B",
                    unit_scale=True,
                    miniters=1,
                    unit_divisor=1024,
                    total=total if total is None else int(total),
            ) as t, path.open("wb") as f:
                block = resp.read(block_size)
                while block:
                    f.write(block)
                    block_num += 1
                    t.update(len(block))
                    block = resp.read(block_num)

            return path.__str__()

    except (KeyboardInterrupt, Exception) as e:
        if path.is_file():
            path.unlink()
        raise e
