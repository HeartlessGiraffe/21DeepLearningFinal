# 21DeepLearningFinal
 ## Train the model
 Run each corresponding python files. Remember to check the files to get argparse information.
 
 ## Test the model
 Download trained models from <https://pan.baidu.com/s/1HDXqzPthM9yObZRFMrlp9A> pw: 7iga
 
 Then put the model dicts into 21DeepLearningFinal/trained_model, change the file name to model.pth and 
 
 ```Bash
 python test_resnet.py -m model_name
 ```
 model_name here can be ResNet18 and ResNet34, corresponding to the model you download. 
 
## Visualize data augmentation 
 ```Bash
 python visualizer.py --help
 ```
and operate according to the helps.
