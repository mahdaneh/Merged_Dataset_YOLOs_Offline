
This repo is a [forked version](https://github.com/ultralytics/yolov3) and aim to implement the method proposed in [Omnia faster r-cnn](https://arxiv.org/abs/1812.02611).
This method is a baseline for generating pseudo-label instances in a merged dataset, which contains many missing-label instances. To read briefly about that,
please read my [blog post](https://mahdaneh.github.io/Blogs/Object_Detector.html).

 Here, instead of faster r-CNN, we used YOLO v3. First, two disjoint datasets are created, using VOC family of datasets, as follows (see `utils/custom_datasets.py`, to load the datasets properly):
  - **VOC 2007**; with following focused categories: <img src="https://render.githubusercontent.com/render/math?math=A">
={*cat, cow, dog, horse, train, sheep*}, the datatset is called VOC7_A.
  - **VOC 2012**; <img src="https://render.githubusercontent.com/render/math?math=B">={*car, motorcycle, bicycle, aeroplane, bus, person*}, the dataset is named VOC12_B.

Then, two YOLOs are trained separately on each of the above datasets in order to be used later for generating pseudo labels for missing label instances from either categories
 <img src="https://render.githubusercontent.com/render/math?math=A">or <img src="https://render.githubusercontent.com/render/math?math=B">.

More precisely, after merging together VOC7_A and VOC12_B, the resultant dataset can contain missing label instances. For example, in the following annotated image from VOC7_A,
 it contains horse annotated, but no annotation for person as "person" category is not in <img src="https://render.githubusercontent.com/render/math?math=A">.

 ![](images/voc7_A.png)

However, after merging VOC7_A and VOC12_B, the person in this image becomes a missing label instance as "person" is a object of interest in <img src="https://render.githubusercontent.com/render/math?math=A\cup B">.

Using YOLO trained on VOC12_B,  the authors proposed to generate pseudo label for the possible missing-label instances from <img src="https://render.githubusercontent.com/render/math?math=B">  in VOC7_A.
 Simialrly, using YOLO trained on VOC7_A, the missing label instance that exist in VOC7_B can be generated.


As main point of start, in order to train a YOLO model, test it,  or generate pseudo_label for missing label instance, `offline_ODs.py` should be used.


## required packages:
- matplotlib
- opencv-contrib-python
- tqdm
- torchvision
- pillow


** Reference **
 Rame, E. Garreau, H. Ben-Younes, and C. Ollion, “Omnia faster r-cnn: Detection in the wild through dataset merging and soft distil-lation,”arXiv preprint arXiv:1812.02611, 2018.
