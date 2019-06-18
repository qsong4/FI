# Fully Multi-head Attention Inference Network
> Fully Multi-head Attention Inference Network(FMAI) is a network base on multi-head attention implement by TF.

## Abstract
FMAI base on matching-aggregation framework, use transformer encoder to extract feature, and use interactive muti-head 
attention do the matching part.(right now its just a toy version but perform well on particular dataset , I will update 
it in the future.)

## Dataset
FMAI version is training on particular dataset which have 2 classes and more than 30m sentences.

## Result
FMAI get acc 0.999 and loss 0.0067 over 20 epoch(little overfitting)

## Tips
