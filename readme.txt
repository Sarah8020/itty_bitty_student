Student IB Model
	See the readme file for this model under its directory.

Teacher Modified TSN Files
	The modified TSN files. The original TSN GitHub repository can be found here - [https://github.com/akaraspt/tinysleepnet]. In the modified TSN, we marked most code we changed from the original with comments, denoted with a double hash ('##').

Android Application
	Demo version: sleepstageapp_safev
	This version of the application demonstrates our pretrained student model executing inference on data that is stored on the device. There is no data transfer. This app will not work as-is - you must store data within the app's data files by using the device file explorer in Android studio. The data file must be in csv format (where each row represents EEG data from 1 epoch), named "test_signals.csv". The data should be stored under data/data/packagename/files. These applications are only meant to demonstrate proof-of-concept, so the code is NOT optimized for performance - therefore, depending on the memory available to the device you are running the applications on, you may have issues with larger files. There is also an upper limit, cutting off data from files containing more than 12 hours of EEG data sampled at 100 Hz.
	
	Bluetooth version: SleepStageApp2.zip
	This version of the application has the same functionality as the demo version. It also includes our attempt at Android BLE functionality. While able to connect to an Earable device, we ran into issues with pairing the device & streaming data. It appears that potentially the device must need be paired in order to transfer data in the Android environment. We were unsuccessful in getting the device to pair, and ultimately it became unfeasible to continue work on the Bluetooth capability given time remaining on the project. To use the BLE application to attempt further progress, you need to alter the device name in the scan settings in MainActivity.java (~line 293), because we set the scan settings to connect to the device by name. Verify your Earable device name and update this setting as necessary.
	Note that you MUST have a physical Android device to run any Bluetooth code - the emulator does not have this capability.
	