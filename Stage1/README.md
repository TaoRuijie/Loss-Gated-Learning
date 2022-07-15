## Stage I: Contrastive Learning

1) Here we use a contrastive learning framework to train the basic speaker encoder. The EER (Vox_O) in our paper is 7.36, we modify a bit recently and now the result is 7.36 in 50 epochs. I believe it can get better if train for more epochs.

2) Any other self-supervised learning framework can be used in Stage I. In my experience, the EER smaller than 7.5 on Vox_O in the first stage is robust to the clustering-training based learning. A bad EER will lead to a bad clustering result, so that Stage II can not work.

3) Notice that our framework contains the [Augmentation adversarial training for unsupervised speaker recognition](https://arxiv.org/pdf/2007.12085.pdf]). Here [AAT](https://github.com/joonson/voxceleb_unsupervised) makes the result better. Our code in this part is also modified based on their project. Thanks for their open-source code!
