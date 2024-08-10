# Itty Bitty Student Sleep Model
## Background
Sleep stage scoring is the process of analyzing PSG (polysomnogram) data to determine the stage of sleep the data represents at any given time interval. Tiny Sleep Net (TSN) is a deep learning model created to automate the process of sleep stage scoring. Tiny Sleep Net is considered "tiny" because it uses a streamlined set of EEG data to determine sleep stages.​

This made TSN the perfect "teacher" model for our "student" model. The Itty Bitty Student model (IB model) uses the same one-channel EEG data to determine sleep stages. With knowledge distillation, we distill knowledge from the accurate but resource-hungry “teacher” model (TSN) to the resource-efficient “student” IB model that can be hosted on a mobile device. The IB model was deployed to a mobile device and demonstrably uses significantly less computational resources as compared to the TSN model, with some loss in accuracy. 

This project was completed as part of a team for my Capstone Design Project with partners [TimFrymire](https://github.com/TimFrymire) and [MuhammadH](https://github.com/MuhammadH).

## Student IB Model
See the [readme](/ib_student/README.md) file for this model under its directory [ib_student](/ib_student/).

## Teacher Modified TSN Files
The modified TSN files are located in the [modified_tsn_final](/modified_tsn_final/) directory. The original TSN GitHub repository can be found here: [tinysleepnet
](https://github.com/akaraspt/tinysleepnet).

In the modified TSN, we marked most code we changed from the original with comments, denoted with a double hash ('##').

## Android Application
### Demo version: [sleepstageapp_safev](/sleepstageapp_safev/)

This version of the application demonstrates our pretrained student model executing inference on data that is stored on the device. There is no data transfer. This app will not work as-is - you must store data within the app's data files by using the device file explorer in Android studio. The data file must be in csv format (where each row represents EEG data from 1 epoch), named "test_signals.csv". The data should be stored under data/data/packagename/files. These applications are only meant to demonstrate proof-of-concept, so the code is NOT optimized for performance - therefore, depending on the memory available to the device you are running the applications on, you may have issues with larger files. There is also an upper limit, cutting off data from files containing more than 12 hours of EEG data sampled at 100 Hz.

### Bluetooth version: [SleepStageApp2](/SleepStageApp2/)

This version of the application has the same functionality as the demo version. It also includes our attempt at Android BLE functionality. While able to connect to an Earable device, we ran into issues with pairing the device & streaming data. It appears that potentially the device must need be paired in order to transfer data in the Android environment. We were unsuccessful in getting the device to pair, and ultimately it became unfeasible to continue work on the Bluetooth capability given time remaining on the project. To use the BLE application to attempt further progress, you need to alter the device name in the scan settings in MainActivity.java (~line 293), because we set the scan settings to connect to the device by name. Verify your Earable device name and update this setting as necessary.

Note that you MUST have a physical Android device to run any Bluetooth code - the emulator does not have this capability.
