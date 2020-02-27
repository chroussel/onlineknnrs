use metrics_core::{Observer, Key};
use dipstick::{AtomicBucket, InputScope, Flush};
use hdrhistogram:: Histogram;

pub struct GraphiteObserver {
    bucket: AtomicBucket
}

impl GraphiteObserver {
    pub fn new(bucket: AtomicBucket) -> GraphiteObserver {
        GraphiteObserver {
            bucket
        }
    }
}

impl Observer for GraphiteObserver {
    fn observe_counter(&mut self, key: Key, value: u64) {
        self.bucket.counter(&key.name()).count(value as usize);
    }

    fn observe_gauge(&mut self, key: Key, value: i64) {
        self.bucket.gauge(&key.name()).value(value);
    }

    fn observe_histogram(&mut self, key: Key, values: &[u64]) {
        let mut histo = Histogram::<u64>::new_with_max(1_000_000_000_000,4).expect("Working histo");
        values.iter().for_each(|h| histo.saturating_record(*h));
        self.bucket.gauge(&format!("{}.count", key.name())).value(values.len());
        self.bucket.gauge(&format!("{}.50", key.name())).value(histo.value_at_quantile(0.50));
        self.bucket.gauge(&format!("{}.90", key.name())).value(histo.value_at_quantile(0.90));
        self.bucket.gauge(&format!("{}.99", key.name())).value(histo.value_at_quantile(0.99));
        self.bucket.gauge(&format!("{}.999", key.name())).value(histo.value_at_quantile(0.999));
        self.bucket.gauge(&format!("{}.mean", key.name())).value(histo.mean());
        self.bucket.gauge(&format!("{}.max", key.name())).value(histo.max());
    }
}