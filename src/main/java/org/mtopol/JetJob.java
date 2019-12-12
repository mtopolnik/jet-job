package org.mtopol;

import com.hazelcast.internal.json.JsonObject;
import com.hazelcast.jet.Jet;
import com.hazelcast.jet.JetException;
import com.hazelcast.jet.JetInstance;
import com.hazelcast.jet.Job;
import com.hazelcast.jet.core.Processor;
import com.hazelcast.jet.pipeline.Pipeline;
import com.hazelcast.jet.pipeline.Sinks;
import com.hazelcast.jet.pipeline.SourceBuilder;
import com.hazelcast.jet.pipeline.SourceBuilder.SourceBuffer;
import com.hazelcast.jet.pipeline.StreamSource;
import com.hazelcast.jet.python.PythonServiceConfig;
import com.hazelcast.jet.server.JetBootstrap;
import com.opencsv.CSVReader;

import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static com.hazelcast.jet.aggregate.AggregateOperations.counting;
import static com.hazelcast.jet.pipeline.WindowDefinition.sliding;
import static com.hazelcast.jet.python.PythonService.mapUsingPython;
import static java.util.Arrays.asList;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.TimeUnit.NANOSECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;

/*
 * Instructions:
 *
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
    private static final long WIN_SIZE = 3;
    private static final long SLIDE_BY = 1;

    private JetInstance jet;
    private static final String MANU_EXAMPLES_BASE = System.getProperty("user.home") + "/dev/python/manu-ml-examples";

    public static void main(String[] args) throws IOException {
        JetJob test = new JetJob();
        test.before();
        try {
            test.sklearn();
        } finally {
            test.after();
        }
    }

    private void before() {
        jet = JetBootstrap.getInstance();
    }

    private void after() {
        Jet.shutdownAll();
    }

    private void sklearn() {
        Pipeline p = Pipeline.create();
        StreamSource<String> source = SourceBuilder
                .stream("income_data", IncomeDataSource::new)
                .fillBufferFn(IncomeDataSource::fillBuffer)
                .destroyFn(IncomeDataSource::destroy)
                .build();
        String skLearn = MANU_EXAMPLES_BASE + "/examples/sklearn";
        PythonServiceConfig pythonServiceConfig = new PythonServiceConfig()
                .setBaseDir(skLearn)
                .setHandlerModule("example_1_inference_jet")
                .setHandlerFunction("handle");
        p.readFrom(source)
         .withoutTimestamps()
         .apply(mapUsingPython(pythonServiceConfig))
         .setLocalParallelism(1)
         .addTimestamps(x -> NANOSECONDS.toMillis(System.nanoTime()), 0)
         .window(sliding(SECONDS.toMillis(WIN_SIZE), SECONDS.toMillis(SLIDE_BY)))
         .aggregate(counting())
         .writeTo(Sinks.logger(wr -> String.format("%,d: %,.1f req/s",
                      MILLISECONDS.toSeconds(wr.end()),
                      (double) wr.result() / WIN_SIZE)
              ));
        Job job = jet.newJob(p);
        Runtime.getRuntime().addShutdownHook(new Thread(job::cancel));
        job.join();
    }

    static class IncomeDataSource {
        private CSVReader csv;
        private String[] columns;
        private final Set<String> intColumns = new HashSet<>(asList(
                "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"
        ));

        IncomeDataSource(Processor.Context ctx) {
            initialize();
        }

        private void initialize() {
            try {
                csv = new CSVReader(new FileReader(MANU_EXAMPLES_BASE + "/datasets/income.data.txt"));
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
                JsonObject json = new JsonObject();
                for (int j = 0; j < values.length; j++) {
                    if (intColumns.contains(columns[j])) {
                        int value = parseOrZero(values[j], values);
                        json.add(columns[j], value);
                    } else {
                        json.add(columns[j], values[j]);
                    }
                }
                buf.add(json.toString());
            }
        }

        void destroy() throws IOException {
            csv.close();
        }
    }

    private static int parseOrZero(String in, String[] row) {
        try {
            return Integer.parseInt(in);
        } catch (NumberFormatException e) {
            System.out.println("Failed to parse '" + in + "' in row: " + Arrays.toString(row));
            return 0;
        }
    }

//    public void benchmark() {
//        AllOfAggregationBuilder<Long> stats = allOfBuilder();
//        Tag<Long> tCount = stats.add(counting());
//        Tag<Long> tMin = stats.add(minBy(comparing(Long::longValue)));
//        Tag<Double> tAvg = stats.add(averagingLong(Long::longValue));
//        Tag<Long> tMax = stats.add(maxBy(comparing(Long::longValue)));
//
//        ServiceFactory<PythonService> pythonServiceFactory = PythonService.factory(
//                "/Users/mtopol/dev/java/hazelcast-jet/hazelcast-jet-python" +
//                        "/src/main/java/com/hazelcast/jet/python/test", "echo", "handle"
//        ).withMaxPendingCallsPerProcessor(8);
//
//        Pipeline p = Pipeline.create();
//        p.readFrom(TestSources.itemStream(4_000_000, (timestamp, seq) -> timestamp))
//         .withNativeTimestamps(0)
//         .map(Object::toString).mapUsingServiceAsyncBatched(pythonServiceFactory,
//                Integer.MAX_VALUE,
//                (PythonService pythonService, List<String> startList1) -> pythonService
//                        .sendRequest(startList1)
//                        .thenApply(startList2 -> {
//                            long now = System.nanoTime();
//                            return startList2
//                                    .stream()
//                                    .map(Long::valueOf)
//                                    .map(start -> NANOSECONDS.toMicros(now - start))
//                                    .collect(toList());
//                        })).setLocalParallelism(8)
//         .window(sliding(SECONDS.toNanos(WIN_SIZE), SECONDS.toNanos(SLIDE_BY)))
//         .aggregate(stats.build())
//         .writeTo(Sinks.logger(wr -> String.format("%,d: %,.1f req/s min %,d µs avg %,.1f µs max %,d µs",
//                 NANOSECONDS.toSeconds(wr.end()),
//                 (double) wr.result().get(tCount) / WIN_SIZE,
//                 wr.result().get(tMin), wr.result().get(tAvg), wr.result().get(tMax))));
//
//        jet.newJob(p).join();
//    }
}