package org.mtopol;

import com.hazelcast.internal.json.JsonObject;
import com.hazelcast.jet.Jet;
import com.hazelcast.jet.JetException;
import com.hazelcast.jet.JetInstance;
import com.hazelcast.jet.Job;
import com.hazelcast.jet.Observable;
import com.hazelcast.jet.config.JobConfig;
import com.hazelcast.jet.core.Processor;
import com.hazelcast.jet.datamodel.WindowResult;
import com.hazelcast.jet.pipeline.Pipeline;
import com.hazelcast.jet.pipeline.Sinks;
import com.hazelcast.jet.pipeline.SourceBuilder;
import com.hazelcast.jet.pipeline.SourceBuilder.SourceBuffer;
import com.hazelcast.jet.pipeline.StreamSource;
import com.hazelcast.jet.python.PythonServiceConfig;
import com.opencsv.CSVReader;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.locks.LockSupport;

import static com.hazelcast.jet.aggregate.AggregateOperations.counting;
import static com.hazelcast.jet.pipeline.WindowDefinition.sliding;
import static com.hazelcast.jet.python.PythonTransforms.mapUsingPython;
import static java.util.Arrays.asList;
import static java.util.concurrent.TimeUnit.SECONDS;

/*
 * Instructions:
 *
 * Build Manu's Model:
 *
 * 1. Pre-requisite is Python 3. On MacOS: brew install python3
 * 2. git clone https://github.com/mtopolnik/manu-ml-examples.git
 * 3. See https://github.com/mtopolnik/manu-ml-examples/wiki/How-to-run
 *    and follow the instructions up to and including the step "Training the Model"
 *
 * Download the Jet Distribution and activate the Python plugin:
 *
 * 1. wget https://github.com/hazelcast/hazelcast-jet/releases/download/v4.2/hazelcast-jet-4.2.tar.gz
 * 2. tar xvf hazelcast-jet-4.2.tar.gz
 * 6. cd hazelcast-jet-4.2
 *    - we'll refer to this directory as $PATH_TO_JET_DISTRO
 * 7. mv opt/hazelcast-jet-python-4.2.jar lib/
 * 8. bin/jet-start
 *
 * Submit the Python ML Job:
 *
 * 1. git clone https://github.com/mtopolnik/jet-job.git
 * 2. Change JetJob.java: make MANU_EXAMPLES_BASE to point to manu-ml-examples
 * 3. mvn clean package
 * 4. $PATH_TO_JET_DISTRO/bin/jet submit -v target/jet-job-1.0-SNAPSHOT.jar
 */
public class JetJob {
    private static final String PYTHON_ML_BASE =
            System.getProperty("user.home") + "/dev/python/manu-ml-examples";

    private static final String INPUT_CSV = "income.data.txt";
    private static final Set<String> INT_COLUMNS = new HashSet<>(asList(
            "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"
    ));
    private static final DateTimeFormatter TIME_FORMATTER = DateTimeFormatter.ofPattern("HH:mm:ss");
    private static final long WIN_SIZE = 3;
    private static final long SLIDE_BY = 1;

    private JetInstance jet;

    public static void main(String[] args) {
        JetJob test = new JetJob();
        test.before();
        try {
            test.sklearn();
        } catch (Throwable t) {
            t.printStackTrace();
        } finally {
            test.after();
        }
    }

    private void before() {
        jet = Jet.bootstrappedInstance();
    }

    private void after() {
        Jet.shutdownAll();
    }

    private void sklearn() {
        String skLearn = PYTHON_ML_BASE + "/examples/sklearn";
        Observable<WindowResult<Long>> reqPerSecond = jet.newObservable();

        StreamSource<String> source = SourceBuilder
                .stream("income_data", IncomeDataSource::new)
                .fillBufferFn(IncomeDataSource::fillBuffer)
                .destroyFn(IncomeDataSource::destroy)
                .build();

        Pipeline p = Pipeline.create();
        p.readFrom(source)
         .withoutTimestamps()
         .rebalance()
         .apply(mapUsingPython(new PythonServiceConfig()
                 .setBaseDir(skLearn)
                 .setHandlerModule("example_1_inference_jet")))
         .setLocalParallelism(1)
         .addTimestamps(x -> System.currentTimeMillis(), 0)
         .window(sliding(SECONDS.toMillis(WIN_SIZE), SECONDS.toMillis(SLIDE_BY)))
         .aggregate(counting())
         .writeTo(Sinks.observable(reqPerSecond));

        reqPerSecond.addObserver(wr -> System.out.format("%s: %,.1f req/s%n",
                TIME_FORMATTER.format(LocalDateTime.ofInstant(Instant.ofEpochMilli(wr.end()), ZoneId.systemDefault())),
                (double) wr.result() / WIN_SIZE));

        Job job = jet.newJob(p, new JobConfig()
                .attachFile(PYTHON_ML_BASE + "/datasets/" + INPUT_CSV));

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            job.cancel();
            LockSupport.parkNanos(500);
            reqPerSecond.destroy();
        }));
        job.join();
    }

    static class IncomeDataSource {
        private final File inputFile;

        private CSVReader csv;
        private String[] columns;

        IncomeDataSource(Processor.Context ctx) {
            inputFile = ctx.attachedFile(INPUT_CSV);
            initialize();
        }

        private void initialize() {
            try {
                csv = new CSVReader(new FileReader(inputFile));
                columns = csv.readNext();
            } catch (IOException e) {
                throw new JetException(e);
            }
        }

        void fillBuffer(SourceBuffer<String> buf) throws IOException {
            for (int i = 0; i < 128; i++) {
                String[] values;
                values = csv.readNext();
                if (values == null) {
                    initialize();
                    values = csv.readNext();
                }
                if (values.length <= 1) {
                    continue;
                }
                buf.add(toJson(columns, values));
            }
        }

        void destroy() throws IOException {
            csv.close();
        }
    }

    static String toJson(String[] keys, String[] values) {
        JsonObject json = new JsonObject();
        for (int j = 0; j < values.length; j++) {
            if (INT_COLUMNS.contains(keys[j])) {
                int value = parseOrZero(values[j], values);
                json.add(keys[j], value);
            } else {
                json.add(keys[j], values[j]);
            }
        }
        return json.toString();
    }

    private static int parseOrZero(String in, String[] row) {
        try {
            return Integer.parseInt(in);
        } catch (NumberFormatException e) {
            System.out.println("Failed to parse '" + in + "' in row: " + Arrays.toString(row));
            return 0;
        }
    }
}
