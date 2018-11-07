# Iterative-Mahine-Teaching-AML-Class


in advanced machine learning class (computer science Universitas Indonesia), I was asked to choose one paper from international conference on machine learning (ICML) in 2017. I chose paper iterative machine teaching [liu et al]. Then surprise I was told to implement what is in the paper that I choose.

iterative machine teaching is a new paradigm that utilizes traditional machine teaching. while machine teaching itself is an inverse problem of machine learning, which is looking for optimum set of data to produce a model similar to the original.

this experiment is a second experiment, because in the first experiment my lecturer judged the method I used (SVM) has selected the data, so no need to be selected again. so for my second experiment I used linear regression.

My experiment was an implementation of an omniscient teacher, using 100 random points and divided equally into two classes. for algorithm implementation, you download or clone this repository, then run experiment.m file

the implementation I did was slightly different from the original paper. I am not looking for the best model by estimating loss function, and not updating the weights with gradient descent. I searched for the best model with learning all the training data and I update the weights with learning from the temporary training data that was chosen iteratively

reference
1. Liu, W., Dai, B., Humayun, A., Tay, C., Yu, C., Smith, L. B., … Song, L. (2017). Iterative Machine Teaching. Retrieved from http://arxiv.org/abs/1705.10470\
2. https://github.com/PRML/PRMLT
