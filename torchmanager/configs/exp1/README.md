# ReRun of svit_3 of DC project on NIH preloaded dataset.

## Changes Introduced:
1. Added conv layer with stride 2 followed by maxpool of stride 2. 
    * **Reason**: feature map dim reduction. otherwise feature maps are so large that passing single sample takes 411MB space.
    * **Effect**: this change reduces feature map to factor of 4.