use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::Hash;

/// RRF smoothing constant (standard default).
const RRF_K: f32 = 60.0;

/// Reciprocal Rank Fusion over any number of best-first ranked id lists.
///
/// `score(id) = Σ_lists 1 / (RRF_K + rank)` (rank 0-based). An empty list is a
/// no-op, so fusion degrades to the surviving leg's order.
#[must_use]
pub fn rrf<I>(lists: &[&[I]], limit: usize) -> Vec<(I, f32)>
where
    I: Eq + Hash + Copy,
{
    let cap = lists.iter().map(|list| list.len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(cap);
    for list in lists {
        for (rank, &id) in list.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let contribution = 1.0 / (RRF_K + rank as f32);
            *scores.entry(id).or_insert(0.0) += contribution;
        }
    }
    let mut fused: Vec<(I, f32)> = scores.into_iter().collect();
    fused.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    fused.truncate(limit);
    fused
}

#[cfg(test)]
mod tests {
    use super::rrf;

    #[test]
    fn overlap_outranks_single_list_hits() {
        let a = [1_u32, 2, 3];
        let b = [3_u32, 4, 5];
        let fused = rrf(&[&a, &b], 10);
        assert_eq!(fused.first().map(|(id, _)| *id), Some(3));
    }

    #[test]
    fn empty_list_is_noop() {
        let a = [10_u32, 20];
        let empty: [u32; 0] = [];
        let fused = rrf(&[&a, &empty], 10);
        assert_eq!(
            fused.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![10, 20]
        );
    }

    #[test]
    fn disjoint_preserves_per_list_order_by_rank() {
        let a = [1_u32];
        let b = [2_u32];
        let fused = rrf(&[&a, &b], 10);
        assert_eq!(fused.len(), 2);
    }
}
