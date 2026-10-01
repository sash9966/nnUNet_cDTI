"""Previous DA5 100-epoch training behavior with compilation disabled.

Use the previous run's ResEnc-M plans to reproduce its architecture, patch size,
and batch size. This class deliberately does not replace those plan settings or
alter augmentation, loss, optimizer, schedule, or foreground sampling.
"""

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import (
    nnUNetTrainerDA5_100epochs,
)


class nnUNetTrainerDA5_100epochsNoCompile(nnUNetTrainerDA5_100epochs):
    def _do_i_compile(self):
        # The base trainer uses this switch for both the network and Dice loss.
        return False
