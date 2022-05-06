package ucdenver.sarahmorgan.sleepstageapp_safev;


import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Chronometer;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.InterpreterFactory;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.text.DecimalFormat;
import java.util.Scanner;

public class MainActivity extends AppCompatActivity {

    //Layout elements:
    Button recordButton;
    Button resultsButton;
    TextView wtime;
    TextView wpercent;
    TextView lstime;
    TextView lspercent;
    TextView dstime;
    TextView dspercent;
    TextView remtime;
    TextView rempercent;
    Chronometer timer;

    //Tensorflow model elements:
    MappedByteBuffer tfliteModel;
    InterpreterApi tflite;

    //Constants:
    int MAX_SIZE = 1440; //30 second epochs, 2 per minute, 720 minutes in 12 hours, 2 epochs * 720 minutes = 1440 max epochs
    int ACTUAL_SIZE = 0; //To be updated once actual size (actual # of epochs) is known.
    int signals_per_epoch = 3000; //30 seconds * 100 Hz = 3000 signals per epoch
    int num_classes = 4; //Stage classes

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
            Log.i("mylog", "No inference input"); //Logging: Indicate null input
        }
        if (output == null) {
            Log.i("mylog", "No inference output"); //Logging: Indicate null output
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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Get references to layout elements:
        recordButton = findViewById(R.id.recordButton);
        resultsButton = findViewById(R.id.resultsButton);
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
        try{
            tfliteModel
                    = FileUtil.loadMappedFile(this,
                    "student_model.tflite");
            tflite = new InterpreterFactory().create(
                    tfliteModel, new InterpreterApi.Options());
        } catch (IOException e){
            Log.i("mylog", "IO Exception: FileUtil student_model.tflite"); //Logging: Indicate IO Exception
        }
        if (tflite == null) {
            Log.i("mylog", "No interpreter"); //Logging: Indicate null interpreter.
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
    }
    @Override
    public void onDestroy() {
        super.onDestroy();
        tflite.close(); //Close resource
    }
}