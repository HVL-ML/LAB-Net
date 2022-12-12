from fastai.vision.all import *
import albumentations as A
from semantic_segmentation_augmentations.holemakerrandom import HoleMakerRandom
from semantic_segmentation_augmentations.cutmixrandom import CutMixRandom
from multispectral import *
        
#### Data augmentation, training #####

class BinaryConverter(DisplayedTransform):
    '''Convert masks.'''
    order=1

    def encodes(self, o:(PILImage)): return o

    def encodes(self, o:(PILMask)):
        o = np.array(o)
        o = np.where(o >0., 1., o)
        return PILMask.create(o)

    
def get_batch_tfms(normalize, size=None):
    batch_tfms=[ Contrast(), Dihedral(), Brightness(), Rotate(max_deg=45), Saturation(),
            Zoom()]
    
    if size is not None: 
         batch_tfms.append(RandomResizedCropGPU(size=size))
            
    batch_tfms.append(Normalize.from_stats(*imagenet_stats))
    
    return batch_tfms


def get_dataloaders(df, x_name, y_name, codes, splitter, bs, 
                    item_tfms, batch_tfms):

    dblock = DataBlock(
                blocks=(ImageBlock(), MaskBlock(codes=codes)),
                splitter=splitter,
                get_x=ColReader(x_name),
                get_y=ColReader(y_name),
                item_tfms=item_tfms,
                batch_tfms=batch_tfms)
    
    dls = dblock.dataloaders(df, bs=bs)
    return dls

def get_cbs(fname, p=0.5, hole_size=(200,200), monitor='valid_loss', min_delta=0.001):

    save_model = SaveModelCallback (monitor=monitor, min_delta=min_delta, fname=fname,with_opt=True)
    show_graph = ShowGraphCallback()
    
    if hole_size!=None:
        cut_mix = CutMixRandom(p=p, hole_maker=HoleMakerRandom(hole_size=hole_size))    
        return [cut_mix, save_model, show_graph]
    
    else:
        return [save_model, show_graph]

    
def fine_tune(learn, epochs, cbs=None, base_lr=1e-3, 
              freeze_epochs=1, lr_mult=100, pct_start=0.3, 
              div=5.0):
    
    print(f"Training for {freeze_epochs} epochs with frozen layers")
    learn.freeze()
    learn.fit_flat_cos(freeze_epochs, slice(base_lr), pct_start=0.99, cbs=cbs)
    base_lr /= 2
    
    print(f"Training unfrozen model for {epochs} epochs")
    
    learn.unfreeze()
    learn.fit_flat_cos(epochs, slice(base_lr/lr_mult, base_lr), 
                      pct_start=pct_start, cbs=cbs)
    
    

class MapAIExperiment():
    def __init__(self, dls, model, metrics, loss_func, epochs, 
                    pretrain, self_attention, act_cls, opt_func, 
                    cutmix):

        self.dls = dls
        self.model = model
        self.metrics = metrics
        self.loss_func = loss_func
        self.epochs = epochs
        self.pretrain = pretrain
        self.self_attention = self_attention
        self.act_cls = act_cls
        self.opt_func = opt_func
        self.cutmix = cutmix




    def fine_tune(self, cbs, base_lr=1e-3, freeze_epochs=1, 
                    lr_mult=100, pct_start=0.3, div=5.0):

        print(f"Training for {freeze_epochs} epochs with frozen layers")
        learn.freeze()
        learn.fit_flat_cos(freeze_epochs, slice(base_lr), pct_start=0.99, cbs=self.cbs)
        base_lr /= 2
        
        print(f"Training unfrozen model for {epochs} epochs")
        
        learn.unfreeze()
        learn.fit_flat_cos(epochs, slice(base_lr/lr_mult, base_lr), 
                        pct_start=pct_start, cbs=self.cbs)

    def train(self):

        
        model_name = str(model).split(' ')[1]

        learn = unet_learner(
                        dls, model, 
                        metrics=self.metrics, loss_func=self.loss_func,
                        pretrained=self.pretrain, 
                        opt_func=self.opt_func,
                        self_attention=self.self_attention
                        ).to_fp16() 

        if pretrain:
            fine_tune(learn, cbs=self.cbs)
        else:
            lr = learn.lr_find(end_lr=1e-2, show_plot=False)
            learn.fit_flat_cos(self.epochs, slice(lr.valley))


# Multispectral #

class SegmentationAlbumentationsTransform(ItemTransform):
#https://www.kaggle.com/code/cordmaur/remotesensing-fastai2-multiband-augmentations/notebook
    split_idx=0 #only training
    def __init__(self, aug, **kwargs): 
        super().__init__(**kwargs)
        self.aug = aug
        
    def encodes(self, x):
        img,mask = x
            
        # for albumentations to work correctly, the channels must be at the last dimension
        aug = self.aug(image=np.array(img.permute(1,2,0)), mask=np.array(mask))
        return TensorImage(aug['image'].transpose(2,0,1)), TensorMask(aug['mask'])

    
    
def get_multiband_item_tfms(size:(list, tuple)=(500,500), p=0.5): 
    item_tfms = SegmentationAlbumentationsTransform(A.Compose([
                                                     A.RandomCrop(size[0], size[1]), 
                                                     A.HorizontalFlip(p=p), 
                                                     A.VerticalFlip(p=p),
                                                     A.RandomBrightnessContrast(p=p), 
                                                     A.RandomRotate90(p=p), 
                                                     A.ShiftScaleRotate(p=p)
                                                    ])) 
    
    return item_tfms




def get_multiband_dataloaders(df, x_name, y_name, codes, splitter, bs, 
                    item_tfms, batch_tfms):

    dblock = DataBlock(
                blocks=(ImageBlock(MImage), MaskBlock(codes=codes)),
                splitter=splitter,
                get_x=ColReader(x_name),
                get_y=ColReader(y_name),
                item_tfms=item_tfms,
                batch_tfms=batch_tfms)
    
    dls = dblock.dataloaders(df, bs=bs)
    return dls
