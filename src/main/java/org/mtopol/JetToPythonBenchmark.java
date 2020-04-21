package org.mtopol;

import com.hazelcast.jet.Jet;
import com.hazelcast.jet.JetInstance;
import com.hazelcast.jet.Job;
import com.hazelcast.jet.aggregate.AllOfAggregationBuilder;
import com.hazelcast.jet.datamodel.Tag;
import com.hazelcast.jet.pipeline.Pipeline;
import com.hazelcast.jet.pipeline.Sinks;
import com.hazelcast.jet.pipeline.test.TestSources;
import com.hazelcast.jet.python.PythonServiceConfig;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.format.DateTimeFormatter;

import static com.hazelcast.function.ComparatorEx.comparing;
import static com.hazelcast.jet.aggregate.AggregateOperations.allOfBuilder;
import static com.hazelcast.jet.aggregate.AggregateOperations.averagingLong;
import static com.hazelcast.jet.aggregate.AggregateOperations.counting;
import static com.hazelcast.jet.aggregate.AggregateOperations.maxBy;
import static com.hazelcast.jet.aggregate.AggregateOperations.minBy;
import static com.hazelcast.jet.pipeline.WindowDefinition.sliding;
import static com.hazelcast.jet.python.PythonTransforms.mapUsingPython;
import static java.util.concurrent.TimeUnit.NANOSECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;

public class JetToPythonBenchmark {
    private static final long WIN_SIZE = 3;
    private static final long SLIDE_BY = 1;
    private static final String ECHO_HANDLER_FUNCTION =
            "def transform_list(input_list):\n" +
            "    return ['reply' + item for item in input_list]\n";

    private JetInstance jet;

    public static void main(String[] args) {
        JetToPythonBenchmark test = new JetToPythonBenchmark();
        test.before();
        try {
            test.benchmark();
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

}
