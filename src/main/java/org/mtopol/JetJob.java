package org.mtopol;

import com.hazelcast.internal.json.JsonObject;
import com.hazelcast.jet.JetException;
import com.hazelcast.jet.JetInstance;
import com.hazelcast.jet.core.Processor;
import com.hazelcast.jet.pipeline.BatchSource;
import com.hazelcast.jet.pipeline.Pipeline;
import com.hazelcast.jet.pipeline.ServiceFactory;
import com.hazelcast.jet.pipeline.Sinks;
import com.hazelcast.jet.pipeline.SourceBuilder;
import com.hazelcast.jet.pipeline.SourceBuilder.SourceBuffer;
import com.hazelcast.jet.python.PythonService;
import com.hazelcast.jet.server.JetBootstrap;
import com.opencsv.CSVReader;

import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import static java.util.Arrays.asList;

public class JetJob {
    public static void main(String[] args) {
        JetInstance jet = JetBootstrap.getInstance();
        Pipeline p = Pipeline.create();
        BatchSource<String> source = SourceBuilder
                .batch("income_data", IncomeDataSource::new)
                .fillBufferFn(IncomeDataSource::fillBuffer)
                .destroyFn(IncomeDataSource::destroy)
                .build();
        ServiceFactory<PythonService> pythonService = PythonService
                .factory("/Users/mtopol/dev/python/manu-ml-examples/examples/sklearn",
                        "example_1_inference_jet", "handle")
                .withMaxPendingCallsPerProcessor(128);
        p.drawFrom(source)
         .mapUsingServiceAsync(pythonService, PythonService::ask).setLocalParallelism(2)
         .drainTo(Sinks.logger());

        jet.newJob(p).join();

    }

    static class IncomeDataSource {
        private final CSVReader csv;
        private final String[] columns;
        private final Set<String> intColumns = new HashSet<>(asList(
                "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"
        ));

        IncomeDataSource(Processor.Context ctx) {
            try {
                csv = new CSVReader(new FileReader(
                        "/Users/mtopol/dev/python/manu-ml-examples/datasets/income.data.txt"));
                columns = csv.readNext();
            } catch (IOException e) {
                throw new JetException(e);
            }
        }

        void fillBuffer(SourceBuffer<String> buf) throws IOException {
            for (int i = 0; i < 128; i++) {
                String[] values = csv.readNext();
                if (values == null) {
                    buf.close();
                    return;
                }
                JsonObject json = new JsonObject();
                for (int j = 0; j < values.length; j++) {
                    if (intColumns.contains(columns[j])) {
                        json.add(columns[j], parseOrZero(values[j]));
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

    private static int parseOrZero(String in) {
        try {
            return Integer.parseInt(in);
        } catch (NumberFormatException e) {
            return 0;
        }
    }
}
