

from torch.utils.data import Dataloader

from AudioDataset import AudioDataset


def main():

    
    daraset = AudioDataset(
        processor=processor,
        model=wav2vec2_model,
        config=config.dataset
    )
    
    dataloader = Dataloader(
        dataset,
        batch_size=config.dataloader.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory
    )





