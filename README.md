# RangeLoss For Gluon

My implement of <Range Loss for Deep Face Recognition with Long-tail> using `MxNet/Gluon`

## Note

- To simplify the problem,the train data that are fed into network can't be shuffled. For example, if your train data of a mini-batch contain 2 classes each with 3 examples, your label must something like this `[1,1,1,4,4,4]`. In order to do that, I implement a simple dataloader called `RangeLossDataLoader`,you can find it in `DataLoader.py`.
- I also test the impact of whether to normalize the output features, the normalized features times a constant value(40 in my test) to scale the norm of features.
- due to my careless, the plot of features training without range loss was plotted on train set while the features training with range loss was plotted on test set. So don't be surprised that the features training  with out range loss look better.

## Image

### SoftMax without normalize features

![1](..\Image\softmax#withoutNorm\6.png)

### SoftMax with normalize features

![2](..\Image\softmax#Norm\5.png)

###Range Loss without normalize features

![3](..\Image\RangeLoss#withoutNorm\5.png)

### Range with normalize features

![4](..\Image\RangeLoss#Norm\6.png)





