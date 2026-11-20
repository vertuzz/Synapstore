//! Iteration 1 acceptance: open a DB file, write and read back 100,000
//! rkyv-archived records — both on raw tables and through the store API.

mod common;

use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata};
use synapstore::archive::{encode, points_into, view};
use synapstore::model::MemoryItem;
use synapstore::store::{EventInput, EventQuery};
use synapstore::tables::{self, EPISODES};
use synapstore::{Config, SynapStore};

const N: u64 = 100_000;

#[test]
fn hundred_thousand_raw_records_roundtrip_zero_copy() {
    let dir = common::TempDir::new("e2e-raw");
    let path = dir.file("records.redb");
    {
        let db = Database::create(&path).unwrap();
        let txn = db.begin_write().unwrap();
        tables::create_all(&txn).unwrap();
        {
            let mut t = txn.open_table(EPISODES).unwrap();
            for i in 0..N {
                let item = MemoryItem {
                    ts: i,
                    id: i * 7 + 1,
                    kind: "k".into(),
                    text: format!("record number {i}"),
                    chunk_id: i,
                };
                t.insert((i, i * 7 + 1), encode(&item).unwrap().as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();
    }
    // Reopen from disk and verify every record via zero-copy views.
    let db = Database::open(&path).unwrap();
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(EPISODES).unwrap();
    assert_eq!(t.len().unwrap(), N);
    let mut expected = 0u64;
    for item in t.range(..).unwrap() {
        let (k, v) = item.unwrap();
        let bytes = v.value();
        let rec = view::<MemoryItem>(bytes).unwrap();
        assert!(points_into(rec, bytes) && points_into(rec.text.as_str().as_bytes(), bytes));
        assert_eq!(k.value(), (expected, expected * 7 + 1));
        assert_eq!(rec.ts.to_native(), expected);
        assert_eq!(rec.text.as_str(), format!("record number {expected}"));
        expected += 1;
    }
    assert_eq!(expected, N);
    let g = t.get((4242, 4242 * 7 + 1)).unwrap().unwrap();
    assert_eq!(view::<MemoryItem>(g.value()).unwrap().chunk_id.to_native(), 4242);
}

#[test]
fn hundred_thousand_events_through_the_store() {
    let dir = common::TempDir::new("e2e-store");
    let path = dir.file("store.redb");
    {
        let s = SynapStore::open(&path, Config::default()).unwrap();
        let batch: Vec<EventInput> = (0..N)
            .map(|i| EventInput {
                kind: "tool_result".into(),
                text: format!("event {i}"),
                timestamp_ms: Some(1_000 + i),
                chunk_id: None,
            })
            .collect();
        for chunk in batch.chunks(10_000) {
            assert_eq!(s.log_events(chunk).unwrap().len(), chunk.len());
        }
        // Idempotent: re-logging the same instant + content is a no-op.
        s.log_events(&batch[..10]).unwrap();
    }
    let s = SynapStore::open(&path, Config::default()).unwrap();
    assert_eq!(s.stats(false).unwrap()["counts"]["episodes"], N);
    let latest = s.recent_events(&EventQuery { limit: 3, ..Default::default() }).unwrap();
    assert_eq!(
        latest.iter().map(|e| e.text.as_str()).collect::<Vec<_>>(),
        vec!["event 99999", "event 99998", "event 99997"]
    );
    let window = s
        .recent_events(&EventQuery {
            limit: 10_000,
            since_ms: Some(1_000 + 500),
            until_ms: Some(1_000 + 599),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(window.len(), 100);
    assert_eq!(window.last().unwrap().text, "event 500");
}
