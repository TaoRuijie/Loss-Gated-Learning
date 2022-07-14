In Stage 2, the baseline and our method shared the same code, I add the `LGL` in the args to simplify the code.

As I mentioned, to make the code clear, this code and the code we used in our paper is a bit different, so the result might be slight different from that in our paper.

* In our paper, we extend the channel size of speaker encoder to 1024 in the iteration 5, in this code we remove this setting to simply the code. You can do that in the last iteration to get the better result. 

* In our paper, we manually determinate the end of each iteration, that is not user-friendly. In this code, we end the iteration if EER can not improve in continuous N = 4 epochs. You can increase it to improve the performance.

I do not have time&resource to run enough epochs for the Stage 2 (I only run 30 epochs get about EER=2.5 already), so if you get the final performance, I will appreciate if you can share your score.txt file with me. Thanks! 
