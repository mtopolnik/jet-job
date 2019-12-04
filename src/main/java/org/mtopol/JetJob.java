package org.mtopol;

import com.hazelcast.internal.json.JsonObject;
import com.hazelcast.jet.Jet;
import com.hazelcast.jet.JetException;
import com.hazelcast.jet.JetInstance;
import com.hazelcast.jet.Job;
import com.hazelcast.jet.core.Processor;
import com.hazelcast.jet.pipeline.Pipeline;
import com.hazelcast.jet.pipeline.ServiceFactory;
import com.hazelcast.jet.pipeline.Sinks;
import com.hazelcast.jet.pipeline.SourceBuilder;
import com.hazelcast.jet.pipeline.SourceBuilder.SourceBuffer;
import com.hazelcast.jet.pipeline.StreamSource;
import com.hazelcast.jet.python.PythonService;
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
import static java.util.Arrays.asList;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.TimeUnit.NANOSECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;


/**
 * Instructions:
 *
 * Pre-requisites
 * 1. Python3. Suggest to install brew install python3
 *
 * Building Manu's Model
 * 1. Clone https://github.com/mtopolnik/manu-ml-examples
 * 2. See https://github.com/mtopolnik/manu-ml-examples/wiki/How-to-run and follow the instructions up and including the Training the Model step.
 *
 *
 * 2. Checkout https://github.com/mtopolnik/manu-ml-examples and change MANU_EXAMPLES_BASE to point to it.
 * Creating a Jet Distribution from Marko's Python Branch and start it
 * 1. Create a Jet distribution from https://github.com/mtopolnik/hazelcast-jet branch python. Use mvn clean install -Pquick
 * 2. cd hazelcast-jet-distribution/target
 * 3. unzip hazelcast-jet-4.0-SNAPSHOT.zip
 * 4. cd into it
 * 5. mv opt/hazelcast-jet-python-4.0-SNAPSHOT.jar lib/
 * 6. bin/jet-start
 *
 * Submit the Python ML Job
 * 1. From another Terminal tab:
 * 2. cd to jet-job (this project)
 * 3. export PATH=$PATH:/Users/gluck/work/marko_ml/hazelcast-jet/hazelcast-jet-distribution/target/hazelcast-jet-4.0-SNAPSHOT/bin
 * 4. mvn clean package
 * 5. jet submit -v target/jet-job-1.0-SNAPSHOT.jar
 *
 *
 *
 */
public class JetJob {
    private static final long WIN_SIZE = 3;
    private static final long SLIDE_BY = 1;

    private JetInstance jet;
    private static final String MANU_EXAMPLES_BASE = System.getProperty("user.home") + "/work/marko_ml/manu-ml-examples";

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

    private void sklearn() {
        Pipeline p = Pipeline.create();
        StreamSource<String> source = SourceBuilder
                .stream("income_data", IncomeDataSource::new)
                .fillBufferFn(IncomeDataSource::fillBuffer)
                .destroyFn(IncomeDataSource::destroy)
                .build();
        String skLearn = MANU_EXAMPLES_BASE + "/examples/sklearn";
        ServiceFactory<PythonService> pythonService = PythonService
                .factory(new PythonServiceConfig()
                        .setBaseDir(skLearn)
                        .setInitScript("init.sh")
                        .setHandlerModule("example_1_inference_jet")
                        .setHandlerFunction("handle"))
                .withMaxPendingCallsPerProcessor(3);
        p.readFrom(source)
         .withoutTimestamps()
         .mapUsingServiceAsyncBatched(pythonService, Integer.MAX_VALUE, PythonService::sendRequest)
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
}