//! HNSW core on raw redb tables: recall vs brute force, idempotency, relinking.

mod common;

use redb::{Database, ReadableDatabase};
use synapstore::hnsw::{self, EntryPoint, HnswParams, View, inspect};
use synapstore::synth::rng::Rng;
use synapstore::tables::{self, HNSW_NODES, META, VECTORS};

fn clustered(rng: &mut Rng, n: usize, dim: usize, clusters: usize) -> Vec<Vec<f32>> {
    let centers: Vec<Vec<f32>> = (0..clusters).map(|_| rng.unit_vector(dim)).collect();
    (0..n).map(|i| rng.perturb(&centers[i % clusters], 0.08)).collect()
}

fn build(db: &Database, p: &HnswParams, vecs: &[Vec<f32>]) {
    let txn = db.begin_write().unwrap();
    tables::create_all(&txn).unwrap();
    {
        let mut v = txn.open_table(VECTORS).unwrap();
        let mut n = txn.open_table(HNSW_NODES).unwrap();
        let mut m = txn.open_table(META).unwrap();
        for (i, x) in vecs.iter().enumerate() {
            hnsw::insert(&mut v, &mut n, &mut m, p, i as u64 + 1, x).unwrap();
        }
    }
    txn.commit().unwrap();
}

#[test]
fn recall_matches_brute_force_and_graph_is_connected() {
    let dir = common::TempDir::new("hnsw");
    let db = Database::create(dir.file("h.redb")).unwrap();
    let mut rng = Rng::new(7);
    let dim = 64;
    let vecs = clustered(&mut rng, 3000, dim, 30);
    let p = HnswParams::default();
    build(&db, &p, &vecs);

    let rtxn = db.begin_read().unwrap();
    let v = rtxn.open_table(VECTORS).unwrap();
    let n = rtxn.open_table(HNSW_NODES).unwrap();
    let m = rtxn.open_table(META).unwrap();
    let ep = EntryPoint::load(&m).unwrap().unwrap();
    let view = View::new(&v, &n);
    let (mut hit, mut total) = (0usize, 0usize);
    for _ in 0..100 {
        let pick = rng.below(vecs.len());
        let q = rng.perturb(&vecs[pick], 0.05);
        let mut visited = 0;
        let got = hnsw::knn(&view, ep, &q, 10, 64, &mut visited).unwrap();
        let want = inspect::brute_force(&v, &q, 10).unwrap();
        let got_ids: Vec<u64> = got.iter().take(10).map(|s| s.id).collect();
        hit += want.iter().filter(|w| got_ids.contains(&w.id)).count();
        total += want.len();
    }
    let recall = hit as f64 / total as f64;
    assert!(recall >= 0.95, "recall@10 = {recall}");
    let st = inspect::index_stats(&n).unwrap();
    assert_eq!(st.nodes, 3000);
    assert_eq!(st.isolated, 0);
    assert_eq!(inspect::reachable_from_entry(&n, Some(ep)).unwrap(), 3000);
    assert!(st.mean_degree[0] > 4.0 && st.mean_degree[0] <= p.m0 as f64);
}

#[test]
fn reinsert_is_idempotent_and_relink_moves_node() {
    let dir = common::TempDir::new("hnsw2");
    let db = Database::create(dir.file("h.redb")).unwrap();
    let mut rng = Rng::new(3);
    let vecs = clustered(&mut rng, 300, 16, 5);
    let p = HnswParams::default();
    build(&db, &p, &vecs);
    let txn = db.begin_write().unwrap();
    {
        let mut v = txn.open_table(VECTORS).unwrap();
        let mut n = txn.open_table(HNSW_NODES).unwrap();
        let mut m = txn.open_table(META).unwrap();
        let o = hnsw::insert(&mut v, &mut n, &mut m, &p, 1, &vecs[0]).unwrap();
        assert!(o.unchanged);
        // Move node 1 onto the vector of node 2: it must now be found there.
        let o = hnsw::insert(&mut v, &mut n, &mut m, &p, 1, &vecs[1]).unwrap();
        assert!(o.relinked && !o.unchanged);
        assert_eq!(tables::meta_get(&m, tables::keys::HNSW_COUNT).unwrap(), Some(300));
        let ep = EntryPoint::load(&m).unwrap().unwrap();
        let view = View::new(&v, &n);
        let mut visited = 0;
        let got = hnsw::knn(&view, ep, &vecs[1], 2, 32, &mut visited).unwrap();
        let ids: Vec<u64> = got.iter().take(2).map(|s| s.id).collect();
        assert!(ids.contains(&1) && ids.contains(&2), "{ids:?}");
    }
    txn.commit().unwrap();
}
