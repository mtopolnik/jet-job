package org.mtopol;

import com.hazelcast.internal.json.JsonObject;
import com.hazelcast.jet.Jet;
import com.hazelcast.jet.JetException;
import com.hazelcast.jet.JetInstance;
import com.hazelcast.jet.Job;
import com.hazelcast.jet.Observable;
import com.hazelcast.jet.aggregate.AllOfAggregationBuilder;
import com.hazelcast.jet.config.JobConfig;
import com.hazelcast.jet.core.Processor;
import com.hazelcast.jet.datamodel.Tag;
import com.hazelcast.jet.datamodel.WindowResult;
import com.hazelcast.jet.pipeline.Pipeline;
import com.hazelcast.jet.pipeline.Sinks;
import com.hazelcast.jet.pipeline.SourceBuilder;
import com.hazelcast.jet.pipeline.SourceBuilder.SourceBuffer;
import com.hazelcast.jet.pipeline.StreamSource;
import com.hazelcast.jet.pipeline.test.TestSources;
import com.hazelcast.jet.python.PythonServiceConfig;
import com.opencsv.CSVReader;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.time.format.FormatStyle;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.TimeZone;
import java.util.concurrent.locks.LockSupport;

import static com.hazelcast.function.ComparatorEx.comparing;
import static com.hazelcast.jet.aggregate.AggregateOperations.allOfBuilder;
import static com.hazelcast.jet.aggregate.AggregateOperations.averagingLong;
import static com.hazelcast.jet.aggregate.AggregateOperations.counting;
import static com.hazelcast.jet.aggregate.AggregateOperations.maxBy;
import static com.hazelcast.jet.aggregate.AggregateOperations.minBy;
import static com.hazelcast.jet.pipeline.WindowDefinition.sliding;
import static com.hazelcast.jet.python.PythonTransforms.mapUsingPython;
import static java.util.Arrays.asList;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.TimeUnit.NANOSECONDS;
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
 * Create a Jet Distribution from Marko's Python branch and start it:
 *
 * 1. git clone https://github.com/mtopolnik/hazelcast-jet.git
 * 2. git checkout python
 * 3. mvn clean install -Pquick
 * 4. cd hazelcast-jet-distribution/target
 * 5. unzip hazelcast-jet-4.0-SNAPSHOT.zip
 * 6. cd hazelcast-jet-4.0-SNAPSHOT
 *    - we'll refer to this directory as $PATH_TO_JET_DISTRO
 * 7. mv opt/hazelcast-jet-python-4.0-SNAPSHOT.jar lib/
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
    private static final Set<String> INT_COLUMNS = new HashSet<>(asList(
            "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"
    ));
    private static final long WIN_SIZE = 3;
    private static final long SLIDE_BY = 1;
    private static final String ECHO_HANDLER_FUNCTION =
            "def transform_list(input_list):\n" +
            "    return ['reply' + item for item in input_list]\n";
    private static final String PYTHON_ML_BASE =
            System.getProperty("user.home") + "/dev/python/manu-ml-examples";
    private static final String INPUT_CSV = "income.data.txt";
    private static final DateTimeFormatter TIME_FORMATTER = DateTimeFormatter.ofPattern("HH:mm:ss");

    private JetInstance jet;

    public static void main(String[] args) {
        JetJob test = new JetJob();
        test.before();
        try {
            test.sklearn();
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
        Observable<WindowResult<Long>> observable = jet.newObservable();

        StreamSource<String> source = SourceBuilder
                .stream("income_data", IncomeDataSource::new)
                .fillBufferFn(IncomeDataSource::fillBuffer)
                .destroyFn(IncomeDataSource::destroy)
                .build();

        Pipeline p = Pipeline.create();
        p.readFrom(source)
         .withoutTimestamps()
         .apply(mapUsingPython(x -> x, new PythonServiceConfig()
                 .setBaseDir(skLearn)
                 .setHandlerModule("example_1_inference_jet")))
         .setLocalParallelism(1)
         .addTimestamps(x -> System.currentTimeMillis(), 0)
         .window(sliding(SECONDS.toMillis(WIN_SIZE), SECONDS.toMillis(SLIDE_BY)))
         .aggregate(counting())
         .writeTo(Sinks.observable(observable));

        JobConfig jobCfg = new JobConfig()
                .attachFile(PYTHON_ML_BASE + "/datasets/" + INPUT_CSV);
        Job job = jet.newJob(p, jobCfg);

        observable.addObserver(wr -> System.out.format("%s: %,.1f req/s%n",
                TIME_FORMATTER.format(LocalDateTime.ofInstant(Instant.ofEpochMilli(wr.end()), ZoneId.systemDefault())),
                (double) wr.result() / WIN_SIZE));

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            job.cancel();
            LockSupport.parkNanos(500);
            observable.destroy();
        }));
        job.join();
    }

    public void benchmark() throws IOException {
        Path echoFile = Files.createTempFile(Paths.get(System.getProperty("user.dir")), "", "echo.py");
        echoFile.toFile().deleteOnExit();
        Files.writeString(echoFile, ECHO_HANDLER_FUNCTION);
        AllOfAggregationBuilder<Long> stats = allOfBuilder();
        Tag<Long> tCount = stats.add(counting());
        Tag<Long> tMin = stats.add(minBy(comparing(Long::longValue)));
        Tag<Double> tAvg = stats.add(averagingLong(Long::longValue));
        Tag<Long> tMax = stats.add(maxBy(comparing(Long::longValue)));

        PythonServiceConfig pythonServiceConfig = new PythonServiceConfig()
                .setHandlerFile(echoFile.toString());

        Pipeline p = Pipeline.create();
        p.readFrom(TestSources.itemStream(4_000_000, (timestamp, seq) -> timestamp))
         .withNativeTimestamps(0)
         .map(Object::toString)
         .apply(mapUsingPython(pythonServiceConfig))
         .map(item -> NANOSECONDS.toMicros(System.nanoTime() - Long.parseLong(item.substring(6))))
         .setLocalParallelism(8)
         .window(sliding(SECONDS.toNanos(WIN_SIZE), SECONDS.toNanos(SLIDE_BY)))
         .aggregate(stats.build())
         .writeTo(Sinks.logger(wr -> String.format("%,d: %,.1f req/s min %,d µs avg %,.1f µs max %,d µs",
                 NANOSECONDS.toSeconds(wr.end()),
                 (double) wr.result().get(tCount) / WIN_SIZE,
                 wr.result().get(tMin), wr.result().get(tAvg), wr.result().get(tMax))));

        Job job = jet.newJob(p);
        Runtime.getRuntime().addShutdownHook(new Thread(job::cancel));
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
