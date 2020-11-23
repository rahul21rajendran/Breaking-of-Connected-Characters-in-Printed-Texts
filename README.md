# merged_character_detection
LTTS
•	The system only works in Linux platform
•	https://drive.google.com/file/d/1px2rlT3Lus3LXBIKKctcpLzl09MtQPXm/view?usp=sharing to download the code and model file
•	Unzip merged_character_detection folder
•	To install the required packages, change current directory to merged_character_detection and run the following command
pip install –r requirements.txt
•	Then run the following command 
git clone https://github.com/myhub/tr.git
cd ./tr
sudo python setup.py install
•	Run the following command to execute the main program, the result will be stored as “Output.png”
 python merged_character_detection.py --image test.png
•	To input other images, change the file name “text.png” in the previous command 
•	Retraining the model:
1)	Manually select and add the images (any size & format) of single character to single_text folder
2)	Manually select and add the images (any size & format) with combined characters to combined_text folder
3)	Run the following command to train the model, the output model file will be stored as “combined_model.h5”
python train_model.py     
To change the number of epochs, open train_model.py and change the number of epoch in line number 114
