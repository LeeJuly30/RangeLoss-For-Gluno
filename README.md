# RangeLoss For Gluon

My implement of `Range Loss for Deep Face Recognition with Long-tail` using `MxNet/Gluon`

## Note

- To simplify the problem,the train data that are fed into network can't be shuffled. For example, if your train data of a mini-batch contain 2 classes each with 3 examples, your label must something like this `[1,1,1,4,4,4]`. In order to do that, I implement a simple dataloader called `RangeLossDataLoader`,you can find it in `DataLoader.py`.
- I also test the impact of whether to normalize the output features, the normalized features times a constant value(40 in my test) to scale the norm of features.
- due to my careless, the plot of features training without range loss was plotted on train set while the features training with range loss was plotted on test set. So don't be surprised that the features training  with out range loss look better.

## Image

### SoftMax without normalize features

<img src="Image/softmax%23withoutNorm/6.png"></img>

### SoftMax with normalize features

<img src="Image/softmax%23Norm/5.png"></img>

### Range Loss without normalize features

<img src="Image/RangeLoss%23withoutNorm/5.png"></img>

### Range Loss with normalize features

<img src="Image/RangeLoss%23Norm/6.png"></img>





