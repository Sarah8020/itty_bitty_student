package ucdenver.sarahmorgan.sleepstageapp;

import android.Manifest;
import android.annotation.SuppressLint;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.BluetoothManager;
import android.bluetooth.BluetoothProfile;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanFilter;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Chronometer;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.InterpreterFactory;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.text.DateFormat;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.Scanner;
import java.util.UUID;

public class MainActivity extends AppCompatActivity {

    //Layout elements:
    Button recordButton;
    Button resultsButton;
    Button btButton;
    TextView wtime;
    TextView wpercent;
    TextView lstime;
    TextView lspercent;
    TextView dstime;
    TextView dspercent;
    TextView remtime;
    TextView rempercent;
    Chronometer timer;

    //Bluetooth elements:
    BluetoothDevice earabledevice = null;
    boolean notscanning = true;
    boolean alock = false;
    BluetoothGattCharacteristic BLE_CMD_RX_CHAR_UUID = null;
    BluetoothGattCharacteristic BLE_EEG_STREAM_CHAR_UUID = null;
    BluetoothGatt btgatt = null;

    //Tensorflow model elements:
    MappedByteBuffer tfliteModel;
    InterpreterApi tflite;

    //Constants:
    int MAX_SIZE = 1440; //30 second epochs, 2 per minute, 720 minutes in 12 hours, 2 epochs * 720 minutes = 1440 max epochs
    int ACTUAL_SIZE = 0; //To be updated once actual size (actual # of epochs) is known.
    int signals_per_epoch = 3000; //30 seconds * 100 Hz = 3000 signals per epoch
    int num_classes = 4; //Stage classes

    /**
     * Saves EEG signal data to .csv format file.
     * This function is not in use in this code - was written
     * intending it to be used to save EEG signal data from BLE
     * device streaming, but we were unsuccessful streaming EEG
     * signal data from BLE device.
     * If data were streamed, and saved in to a 2D array of floats
     * with the given structure, this function would allow data to be
     * saved in a file for immediate or later inference. It has been
     * verified to be compatible with getData().
     * @param data  2D array of floats; represents EEG signal data:
     *              shape = [# of epochs][signals_per_epoch]
     * @return      File object representing the file data was just saved in
     * @throws IOException  If file operations fail.
     */
    public File saveData(float[][] data) throws IOException {
        //Set filename as current date & time:
        Date recording_date = Calendar.getInstance().getTime(); //Get the current date & time
        @SuppressLint("SimpleDateFormat") DateFormat df = new SimpleDateFormat("yyyy-mm-dd hh:mm:ss"); //Format the date & time
        String filename = df.format(recording_date) + ".csv"; //Set filename

        //Create file:
        File file = new File(getApplicationContext().getFilesDir(), filename);
        if (!file.exists()) {
            file.createNewFile(); //Returns false if the filename already exists
            Log.i("mylog", "File Created at:");
            Log.i("mylog", file.getPath()); //Logging: Indicate file location
        }

        //Create FileWriter object:
        FileWriter writer = null;
        try {
            writer = new FileWriter(file);
        } catch (IOException e) {
            e.printStackTrace();
        }

        //Write data to file:
        int rows = data.length;
        int rows_len = data[0].length;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < rows_len; j++) {
                String num = Float.toString(data[i][j]); //Convert each number to string
                writer.write(num); //Write string to file using FileWriter object
                writer.write(","); //CSV format
            }
            writer.write("\n"); //Newline at the end of every row (each row is one epoch of sleep signals)
        }
        writer.close();
        return file; //Return the File object
    } //End saveData() function

    /**
     * Reads EEG signal data from .csv format file.
     * @param file  File object representing the file data should be read from
     * @return      3D array of floats; represents EEG signal data (read from file):
     *              shape = [# of epochs][signals_per_epoch][# of channels] = [ACTUAL_SIZE][signals_per_epoch][1]
     * @throws FileNotFoundException
     */
    public float[][][] getData(File file) throws FileNotFoundException {
        //Create Scanner object:
        Scanner scanner = null;
        try {
            scanner = new Scanner(file);
        } catch (IOException e) {
            Log.i("mylog", "IO Exception: getData Scanner"); //Logging: Indicate IO Exception
        }

        //Read string data from file:
        int i = 0;
        String data_strings[][] = new String[MAX_SIZE][signals_per_epoch];
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine(); //Read a line
            String[] row = line.split(","); //Split by delimiter into array of strings
            data_strings[i] = row; //Save the row data into the 2D array
            i += 1; //Increment the row #
        }
        scanner.close();

        //Convert string data to float:
        ACTUAL_SIZE = i; //Set actual size (actual # of epochs)
        float data[][][] = new float[ACTUAL_SIZE][signals_per_epoch][1]; //Add channel dimension for model CNN
        for (i = 0; i < ACTUAL_SIZE; i ++) {
            for (int j = 0; j < signals_per_epoch; j++) {
                data[i][j][0] = Float.parseFloat(data_strings[i][j]);
            }
        }
        return data; //Return 3D array of EEG data
    } //End getData() function

    /**
     * Performs inference on provided EEG data.
     * @param tflite tf lite model
     * @param file   file containing EEG data
     * @return       model output; 2D array of predictions:
     *               shape = [epoch #][# of classes, largest value indicates predicted class (index = class)]
     * @throws FileNotFoundException
     */
    public float[][] doInference(InterpreterApi tflite, File file) throws FileNotFoundException {
        //Input shape:
        float[][][] input = getData(file);

        //Output shape:
        float[][] output = new float[ACTUAL_SIZE][num_classes];

        //Logging: Indicate input shape
        Log.i("mylog", "Input rows: " + input.length);
        Log.i("mylog", "Input columns: " + input[0].length);

        if (input == null) {
            Log.i("mylog", "No input."); //Logging: Indicate null input
        }
        if (output == null) {
            Log.i("mylog", "No output."); //Logging: Indicate null output
        }
        long start = System.nanoTime(); //Measuring time to produce predictions
        tflite.run(input, output); //Run inference
        long end = System.nanoTime(); //Measuring time to produce predictions
        long duration = (end - start); //Time in nanoseconds
        Log.i("mylog", "Time to produce predictions: " + Long.toString(duration)); //Logging: Indicate time to produce predictions
        return output; //Return 2D array of predictions
    } //End doInference() function

    /**
     * Process inference results (process predictions).
     * @param tflite    tf lite model
     * @param file      file containing EEG data
     * @throws FileNotFoundException
     */
    public void processInference(InterpreterApi tflite, File file) throws FileNotFoundException {
        //Do inference:
        float[][] output = doInference(tflite, file);

        //Logging: Indicate output shape
        Log.i("mylog", "Output rows: " + output.length);
        Log.i("mylog", "Output columns: " + output[0].length);

        //Argmax approximation:
        int labels[] = new int[ACTUAL_SIZE];
        int max = 0;
        for (int i = 0; i < ACTUAL_SIZE; i++) {
            for (int j = 0; j < num_classes; j++) {
                //Get max prediction value for each epoch; index indicates predicted class:
                //    output[i][j] > output[i][max] ? --> true, max = j; false, max = max
                max = output[i][j] > output[i][max] ? j : max;
            }
            labels[i] = max; //Save predicted class label
        }

        //Get epoch sums per class (total epochs predicted to belong to each class):
        float epoch_sums_per_class[] = new float[num_classes];
        for (int i = 0; i < ACTUAL_SIZE; i++) {
            epoch_sums_per_class[labels[i]] += 1;
        }

        //Process inference results into percentages & time values and post to layout:
        float[] c = new float[num_classes];
        float[] p = new float[num_classes];
        float sum_all_classes = 0;
        for (int i = 0; i < num_classes; i++) {
            Log.i("mylog", "Per class sum: " + Integer.toString(i) + ": " + Float.toString(epoch_sums_per_class[i])); //Logging: Indicate epoch sums per class
            sum_all_classes += epoch_sums_per_class[i];
        }
        for (int i = 0; i < num_classes; i++) {
            p[i] = (float) ((epoch_sums_per_class[i] / sum_all_classes) * 100.00); //Percentage
            c[i] = (float) ((epoch_sums_per_class[i] * 30.0) / 60.0); //# epochs to minutes
        }

        DecimalFormat df = new DecimalFormat("##.##"); //Format percentages
        wpercent.setText(String.valueOf(df.format(p[0])) + "%");
        lspercent.setText(String.valueOf(df.format(p[1])) + "%");
        dspercent.setText(String.valueOf(df.format(p[2])) + "%");
        rempercent.setText(String.valueOf(df.format(p[3]) + "%"));
        wtime.setText(String.valueOf(c[0]) + " Min.");
        lstime.setText(String.valueOf(c[1]) + " Min.");
        dstime.setText(String.valueOf(c[2]) + " Min.");
        remtime.setText(String.valueOf(c[3]) + " Min.");
    } //End processInference() function

    /* Bluetooth code: */
    //onActivityResult() is a callback for asking the user to enable Bluetooth permissions.
    protected void onActivityResult(int requestCode, int resultCode, Intent intent) {
        super.onActivityResult(requestCode, resultCode, intent);
        if (requestCode == 1) {
            if (resultCode == RESULT_OK) {
                Log.i("mylog", "Bluetooth enabled "); //Logging: Indicate Bluetooth enabled
            } else {
                Log.i("mylog", "Bluetooth not enabled"); //Logging: Indicate Bluetooth not enabled
            }
        }
    }

    //bluetoothScan() is a custom function to scan for Bluetooth device.
    @SuppressLint("MissingPermission")
    public void bluetoothScan() {
        BluetoothManager btmanager = getSystemService(BluetoothManager.class);
        BluetoothAdapter btadapter = btmanager.getAdapter();
        BluetoothLeScanner blescanner = btadapter.getBluetoothLeScanner();

        if (blescanner != null) {
            //Match device by name ONLY and match ONE DEVICE ONLY:
            List<ScanFilter> filters = new ArrayList<>();
            ScanFilter filter = new ScanFilter.Builder().setDeviceName("EB_DEV6865").build(); //You will need to change this name to connect to another Earable device!!
            filters.add(filter);
            ScanSettings.Builder settingsBuilder = new ScanSettings.Builder();
            settingsBuilder.setCallbackType(ScanSettings.CALLBACK_TYPE_FIRST_MATCH);
            ScanSettings settings = settingsBuilder.build();

            blescanner.startScan(filters, settings, scancallback); //Start scanning
            notscanning = false;
            Log.i("mylog", "Scanning"); //Logging: Indicate scanning begun

            final Handler handler = new Handler(Looper.getMainLooper());
            handler.postDelayed(new Runnable() {
                @Override
                public void run() {
                    blescanner.stopScan(scancallback); //Stop scanning after 5 seconds
                    notscanning = true;
                    Log.i("mylog", "Not scanning"); //Logging: Indicate scanning ended
                }
            }, 5000);
        } else {
            Log.i("mylog", "Scanner null"); //Logging: Indicate scanner null
        }
    }

    //ScanCallback() is the callback for startScan().
    private ScanCallback scancallback = new ScanCallback() {
        @SuppressLint("MissingPermission")
        @Override
        public void onScanResult(int callbackType, ScanResult result) {
            super.onScanResult(callbackType, result);
            Log.i("mylog", result.toString()); //Logging: Indicate result of scan
            earabledevice = result.getDevice();
            earabledevice.connectGatt(getApplicationContext(), false, gattcallback); //Connect to device
        }
        @SuppressLint("MissingPermission")
        @Override
        public void onScanFailed(int errorCode) {
            super.onScanFailed(errorCode);
            Log.i("mylog", "Scan failed - error code: " + Integer.toString((errorCode))); //Logging: Indicate scan fail error code
            //Ensure user settings are correct - notify them with Toast:
            Toast toast = Toast.makeText(getApplicationContext(), "Bluetooth Error: Must allow app to use location services in settings & enable Bluetooth to use Bluetooth. Check settings & restart app.", Toast.LENGTH_LONG);
            toast.show();
            BluetoothAdapter.getDefaultAdapter().disable(); //Disable adapter in case of failed scan
        }
    };

    //BluetoothGattCallback() is the callback for connectGatt().
    private BluetoothGattCallback gattcallback = new BluetoothGattCallback() {
        @Override
        public void onPhyUpdate(BluetoothGatt gatt, int txPhy, int rxPhy, int status) {
            super.onPhyUpdate(gatt, txPhy, rxPhy, status);
            Log.i("mylog", "onPhyUpdate called"); //Logging: Indicate callback
        }
        @Override
        public void onPhyRead(BluetoothGatt gatt, int txPhy, int rxPhy, int status) {
            super.onPhyRead(gatt, txPhy, rxPhy, status);
            Log.i("mylog", "onPhyRead called"); //Logging: Indicate callback
        }
        @SuppressLint("MissingPermission")
        @Override
        public void onConnectionStateChange(BluetoothGatt gatt, int status, int newState) {
            super.onConnectionStateChange(gatt, status, newState);
            if (newState == BluetoothProfile.STATE_CONNECTED) { //Connected to GATT server (BLE device)
                Log.i("mylog", "Connected to BLE device"); //Logging: Indicate connection made
                Log.i("mylog", "Status: " + status); //Logging: Indicate status of connection
                Log.i("mylog", "State: " + newState); //Logging: Indicate state of connection
                btgatt = gatt; //Save gatt
                gatt.getDevice().createBond(); //Pair device
                //gatt.discoverServices(); //Discover services
            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                Log.i("mylog", "BLE disconnected"); //Logging: Indicate connection disconnected
                Log.i("mylog", "Status: " + status); //Logging: Indicate status of connection
                Log.i("mylog", "State: " + newState); //Logging: Indicate state of connection
                gatt.close(); //Close gatt
            }
        }
        @SuppressLint("MissingPermission")
        @Override
        public void onServicesDiscovered(BluetoothGatt gatt, int status) {
            super.onServicesDiscovered(gatt, status);
            if (status == gatt.GATT_SUCCESS) {
                List<BluetoothGattService> services = gatt.getServices(); //Get list of services
                if (services.isEmpty()) {
                    Log.i("mylog", "No services available"); //Logging: Indicate no services available
                    return;
                }
                for (int i = 0; i < services.size(); i++) {
                    Log.i("mylog", "Service:");
                    if (services.get(i).getUuid().equals(UUID.fromString("d0611e78-bbb4-4591-a5f8-487910ae4366"))) { // Ignore continuity service
                        continue;
                    }
                    Log.i("mylog", (services.get(i).getUuid().toString())); //Logging: Indicate service UUIDs
                    List<BluetoothGattCharacteristic> characteristics = services.get(i).getCharacteristics();
                    if (!characteristics.isEmpty()) {
                        Log.i("mylog", "Characteristics:");
                        for (i = 0; i < characteristics.size(); i++) {
                            Log.i("mylog", (characteristics.get(i).getUuid().toString())); //Logging: Indicate characteristic UUIDs
                            if (characteristics.get(i).getUuid().equals(UUID.fromString("45420101-0000-ffff-ff45-415241424c45"))) {
                                BLE_CMD_RX_CHAR_UUID = characteristics.get(i); //BLE start data streaming characteristic
                            } else if (characteristics.get(i).getUuid().equals(UUID.fromString("45420201-0000-ffff-ff45-415241424c45"))) {
                                BLE_EEG_STREAM_CHAR_UUID = characteristics.get(i); //BLE EEG stream characteristic
                            }
                        }
                    }
                }
            } else {
                Log.i("mylog", "onServicesDiscovered received: " + status); //Logging: Indicate status if services not discovered
            }

            if (BLE_EEG_STREAM_CHAR_UUID != null) {
                Log.i("mylog", "Subscribing to notifications on EEG characteristic..."); //Logging: Indicate subscribing to notifications
                BluetoothGattDescriptor descriptor = BLE_EEG_STREAM_CHAR_UUID.getDescriptor(BLE_EEG_STREAM_CHAR_UUID.getUuid());
                gatt.setCharacteristicNotification(BLE_EEG_STREAM_CHAR_UUID, true);
                descriptor.setValue(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
                gatt.writeDescriptor(descriptor);
            }
            if (BLE_CMD_RX_CHAR_UUID != null) {
                Log.i("mylog", "Sending command to start data streaming..."); //Logging: Indicate attempting to write command to start data streaming

                //Code to send time sync command (does not appear to be necessary, in any case, always received back a code indicating GATT_WRITE_NOT_PERMITTED):
                /*int unixTime = (int) (System.currentTimeMillis() / 1000);
                byte[] timesync = new byte[]{
                        (byte) 0x07,
                        (byte) (unixTime >> 24),
                        (byte) (unixTime >> 16),
                        (byte) (unixTime >> 8),
                        (byte) unixTime
                };
                Log.i("mylog", "Time stamp: " + Long.toString(unixTime)); //Logging: Indicate time stamp
                BLE_CMD_RX_CHAR_UUID.setWriteType(2);
                BLE_CMD_RX_CHAR_UUID.setValue(timesync);
                gatt.writeCharacteristic(BLE_CMD_RX_CHAR_UUID);*/

                BLE_CMD_RX_CHAR_UUID.setWriteType(2);
                BLE_CMD_RX_CHAR_UUID.setValue(new byte[]{(byte) 0x01, (byte) 0xff, (byte) 0x03}); //Command to start data streaming
                gatt.writeCharacteristic(BLE_CMD_RX_CHAR_UUID); //Write value to characteristic - always received back a code indicating GATT_WRITE_NOT_PERMITTED
            }
        }
        @Override
        public void onCharacteristicRead(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic, int status) {
            super.onCharacteristicRead(gatt, characteristic, status);
            alock = false;
            Log.i("mylog", "onCharacteristicRead called for " + characteristic.getUuid().toString()); //Logging: Indicate callback characteristic
            Log.i("mylog", "onCharacteristicRead status: " + status);  //Logging: Indicate read status
            if (status == gatt.GATT_SUCCESS) {
                Log.i("mylog", "onCharacteristicRead success for " + characteristic.getUuid().toString()); //Logging: Indicate success for callback characteristic
            }
        }
        @SuppressLint("MissingPermission")
        @Override
        public void onCharacteristicWrite(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic, int status) {
            super.onCharacteristicWrite(gatt, characteristic, status);
            Log.i("mylog", "onCharacteristicWrite called for " + characteristic.getUuid().toString()); //Logging: Indicate callback characteristic
            Log.i("mylog", "onCharacteristicWrite called " + status); //Logging: Indicate write status
            if (status == gatt.GATT_SUCCESS) {
                Log.i("mylog", "onCharacteristicWrite success for " + characteristic.getUuid().toString()); //Logging: Indicate success for callback characteristic
            }
            //while(!gatt.readCharacteristic(BLE_EEG_STREAM_CHAR_UUID)){} //Read until read is success (attempted)
        }
        @Override
        public void onCharacteristicChanged(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic) {
            super.onCharacteristicChanged(gatt, characteristic);
            Log.i("mylog", "onCharacteristicChanged called for " + characteristic.getUuid().toString()); //Logging: Indicate callback characteristic
        }
        @Override
        public void onDescriptorRead(BluetoothGatt gatt, BluetoothGattDescriptor descriptor, int status) {
            super.onDescriptorRead(gatt, descriptor, status);
            Log.i("mylog", "onDescriptorRead called"); //Logging: Indicate callback
        }
        @Override
        public void onDescriptorWrite(BluetoothGatt gatt, BluetoothGattDescriptor descriptor, int status) {
            super.onDescriptorWrite(gatt, descriptor, status);
            Log.i("mylog", "onDescriptorWrite called"); //Logging: Indicate callback
        }
        @Override
        public void onReliableWriteCompleted(BluetoothGatt gatt, int status) {
            super.onReliableWriteCompleted(gatt, status);
            Log.i("mylog", "onReliableWriteCompleted called"); //Logging: Indicate callback
        }
        @Override
        public void onReadRemoteRssi(BluetoothGatt gatt, int rssi, int status) {
            super.onReadRemoteRssi(gatt, rssi, status);
            Log.i("mylog", "onReadRemoteRssi called"); //Logging: Indicate callback
        }
        @Override
        public void onMtuChanged(BluetoothGatt gatt, int mtu, int status) {
            super.onMtuChanged(gatt, mtu, status);
            Log.i("mylog", "onMtuChanged called"); //Logging: Indicate callback
        }
        @Override
        public void onServiceChanged(@NonNull BluetoothGatt gatt) {
            super.onServiceChanged(gatt);
            Log.i("mylog", "onServiceChanged called"); //Logging: Indicate callback
        }
    };

    //BroadcastReceiver() is the callback for createBond() to pair a device.
    private final BroadcastReceiver bondStateReceiver = new BroadcastReceiver() {
        @SuppressLint("MissingPermission")
        @Override
        public void onReceive(Context context, Intent intent) {
            final String action = intent.getAction();
            final BluetoothDevice device = intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE);

            //Take action depending on new bond state:
            if (action.equals(BluetoothDevice.ACTION_BOND_STATE_CHANGED)) {
                final int bondState = intent.getIntExtra(BluetoothDevice.EXTRA_BOND_STATE, BluetoothDevice.ERROR);
                switch (bondState) {
                    case BluetoothDevice.BOND_BONDING:
                        Log.i("mylog", "BONDING"); //Logging: Indicate bond state
                        break;
                    case BluetoothDevice.BOND_BONDED:
                        Log.i("mylog", "BONDED"); //Logging: Indicate bond state
                        btgatt.discoverServices(); //Discover services
                        break;
                    case BluetoothDevice.BOND_NONE:
                        Log.i("mylog", "NO BOND"); //Logging: Indicate bond state
                        break;
                }
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Get references to layout elements:
        recordButton = findViewById(R.id.recordButton);
        resultsButton = findViewById(R.id.resultsButton);
        btButton = findViewById(R.id.btButton);
        wtime = findViewById(R.id.wtimeTextView);
        wpercent = findViewById(R.id.wpercentTextView);
        lstime = findViewById(R.id.lstimeTextView);
        lspercent = findViewById(R.id.lspercentTextView);
        dstime = findViewById(R.id.dstimeTextView);
        dspercent = findViewById(R.id.dspercentTextView);
        remtime = findViewById(R.id.remtimeTextView);
        rempercent = findViewById(R.id.rempercentTextView);
        timer = findViewById(R.id.timerChronometer);

        //Initialise the tflite model:
        try {
            tfliteModel
                    = FileUtil.loadMappedFile(this,
                    "student_model.tflite");
            tflite = new InterpreterFactory().create(
                    tfliteModel, new InterpreterApi.Options());
        } catch (IOException e) {
            Log.i("mylog", "IO Exception: FileUtil student_model.tflite"); //Logging: Indicate IO Exception
        }
        if (tflite == null) {
            Log.i("mylog", "No interpreter."); //Logging: Indicate null interpreter.
        }

        recordButton.setOnClickListener(new View.OnClickListener() {
            boolean recording = false;
            @Override
            public void onClick(View view) {
                if (recording == false) { //Start recording
                    recordButton.setBackgroundColor(getResources().getColor(R.color.record_red)); //On click make button color change
                    recordButton.setText(R.string.recordButton2); //On click make button text change
                    timer.setBase(SystemClock.elapsedRealtime()); //Reset chronometer
                    timer.start(); //Start chronometer
                    timer.setFormat("Recording... %s"); //Set chronometer format
                    //Reset the results:
                    wpercent.setText(R.string.blank);
                    lspercent.setText(R.string.blank);
                    dspercent.setText(R.string.blank);
                    rempercent.setText(R.string.blank);
                    wtime.setText(R.string.blank);
                    lstime.setText(R.string.blank);
                    dstime.setText(R.string.blank);
                    remtime.setText(R.string.blank);
                } else if (recording == true) { //Stop recording
                    recordButton.setBackgroundColor(getResources().getColor(R.color.blue)); //On click make button color change
                    recordButton.setText(R.string.recordButton); //On click make button text change
                    timer.setBase(SystemClock.elapsedRealtime()); //Reset chronometer
                    timer.stop(); //Stop chronometer
                    timer.setText(R.string.stoppedrecording); //Set chronometer text to indicate recording stopped
                }
                recording = !recording; //Change recording status after every button click
            }
        });

        resultsButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                File file = new File(getApplicationContext().getFilesDir(), "test_signals.csv"); //Create file object from test signals file
                try {
                    processInference(tflite, file); //Process inference
                    Toast toast = Toast.makeText(getApplicationContext(), "Inference Completed", Toast.LENGTH_SHORT);
                    toast.show();
                } catch (FileNotFoundException e) {
                    Log.i("mylog", "Test signals file not found"); //Logging: Indicate file not found
                }
            }
        });

        btButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.i("mylog", "Bt button click"); //Logging: Indicate Bluetooth button click
                //Bluetooth:
                BluetoothManager btmanager = getSystemService(BluetoothManager.class);
                BluetoothAdapter btadapter = btmanager.getAdapter();
                if (btmanager == null) {
                    Log.i("mylog", "Bluetooth not supported"); //Logging: Indicate Bluetooth not supported
                }
                //Enable Bluetooth (ask user for permission):
                if (!btadapter.isEnabled()) {
                    Intent enablebtIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
                    startActivityForResult(enablebtIntent, 1);
                    if (ActivityCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED) {
                        Log.i("mylog", "Bluetooth not enabled (permission denied by user)"); //Logging: Indicate Bluetooth not enabled
                    }
                }
                bluetoothScan(); //Initiate Bluetooth scan
            }
        });
    }
    @Override
    public void onDestroy() {
        super.onDestroy();
        tflite.close(); //Close resource
    }
}