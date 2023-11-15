# import image
import os

from cellbin.image import Image
from cellbin.modules.cell_segmentation import CellSegmentation

from stereo.image.segmentation.seg_utils.base_cell_seg_pipe.cell_seg_pipeline import CellSegPipe
from stereo.log_manager import logger


class CellSegPipeV1Pro(CellSegPipe):

    def run(self):
        logger.info('Start do cell mask, this will take some minutes.')
        cell_seg = CellSegmentation(
            model_path=self.model_path,
            gpu=self.kwargs.get('gpu', '-1'),
            num_threads=self.kwargs.get('num_threads', 0),
        )
        logger.info(f"Load {self.model_path}) finished.")

        image = Image()
        image.read(image=self.img_path)

        # Run cell segmentation
        mask = cell_seg.run(image.image)
        self.mask = mask

        self.save_cell_mask()

    def save_cell_mask(self):
        cell_mask_path = os.path.join(self.out_path, f"{self.file_name[-1]}_mask.tif")
        Image.write_s(self.mask, cell_mask_path, compression=True)
        logger.info('Result saved : %s ' % (cell_mask_path))
